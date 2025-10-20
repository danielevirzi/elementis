"""
RAG pipeline for chunking cleaned markdown documents and storing in ChromaDB

Pipeline:
1. Extract markdown from PDFs using Docling (extract_markdown.py)
2. Manually clean markdown files
3. Chunk markdown using LlamaIndex (this module)
4. Encode and store chunks in ChromaDB (this module)
5. Hybrid retrieval: Semantic + Keyword search
6. Answer generation with Granite LLM
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import torch
from chromadb.config import Settings
from keybert import KeyLLM
from keybert.llm import TextGeneration
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import util
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

from utils import load_config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for processing cleaned markdown documents
    
    Pipeline workflow:
    1. Extract markdown from PDFs using Docling (separate script)
    2. Manually clean markdown files in data/processed/
    3. Load cleaned markdown → chunk with LlamaIndex → store in ChromaDB
    
    This class handles steps 3-4 (chunking and vector storage)
    """
    
    def __init__(self):
        """Initialize RAG pipeline"""
        logger.info("Initializing RAG Pipeline...")
        
        # Load configuration
        self.config = load_config().get("rag", {})
        
        # Paths
        self.documents_path = Path(os.getenv("DOCUMENTS_PATH", "./data/documents"))
        self.processed_path = Path(os.getenv("PROCESSED_PATH", "./data/processed"))
        self.db_path = Path(os.getenv("CHROMA_DB_PATH", "./data/vector_db"))
        
        # Ensure directories exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Use GPU for embedding model if available (we have 2.5GB headroom)
        # This should speed up RAG retrieval by encoding queries faster
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Embedding model will use: GPU (we have memory headroom)")
        else:
            self.device = "cpu"
            logger.info(f"Embedding model will use: CPU (no GPU available)")
        
        # Initialize Docling document converter with simpler configuration
        # Only needed for PDF extraction - commented out when working with pre-processed markdown
        # self.doc_converter = DocumentConverter()
        # logger.info("Docling document converter initialized")
        
        # Initialize embedding model with IBM Granite multilingual model
        # Better for Italian documents
        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", 
            "ibm-granite/granite-embedding-278m-multilingual"
        )
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        # Load embedding model using LlamaIndex wrapper (loads only once)
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            device=self.device
        )
        logger.info(f"LlamaIndex embedding model initialized on {self.device}")
        
        # Access the underlying SentenceTransformer model for direct use
        # HuggingFaceEmbedding stores the model in _model attribute (private)
        try:
            # Try to access the internal model
            if hasattr(self.embed_model, '_model'):
                self.sentence_transformer = self.embed_model._model
            elif hasattr(self.embed_model, 'model'):
                self.sentence_transformer = self.embed_model.model
            else:
                # Fallback: create a separate instance but only for encoding operations
                logger.warning("Could not access internal model, creating shared reference")
                self.sentence_transformer = self.embed_model
        except Exception as e:
            logger.warning(f"Could not access internal model: {e}, using embed_model directly")
            self.sentence_transformer = self.embed_model
            
        logger.info(f"Embedding model ready (single instance)")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "elementis_documents")
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Elementis document collection"}
        )
        
        # Initialize ChromaDB vector store for LlamaIndex
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize node parser (chunking strategy)
        chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(f"Node parser configured: chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Initialize index (will be loaded/created later)
        self.index = None
        
        # Initialize reranker model (Jina reranker for better retrieval)
        self.reranker_model = None
        self.reranker_tokenizer = None
        self._init_reranker()
        
        # Initialize Granite LLM for answer generation
        self.llm_model = None
        self.llm_tokenizer = None
        self._init_llm()
        
        # Initialize KeyLLM for keyword extraction
        self.keyllm_model = None
        self._init_keyllm()
        
        logger.info(f"ChromaDB collection '{collection_name}' ready")
        logger.info("RAG Pipeline initialized successfully")
    
    def _init_reranker(self):
        """Initialize Jina reranker model on GPU for better retrieval quality"""
        try:
            reranker_model_path = os.getenv(
                "RERANKER_MODEL", 
                "jinaai/jina-reranker-v2-base-multilingual"
            )
            
            logger.info(f"Loading Jina reranker: {reranker_model_path}")
            
            # Load reranker on GPU if available
            if torch.cuda.is_available():
                logger.info("Loading reranker on GPU...")
                
                # Clear cache before loading
                torch.cuda.empty_cache()
                self._check_gpu_memory("Reranker loading")
                
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                    reranker_model_path,
                    trust_remote_code=True
                )
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    reranker_model_path,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.reranker_device = "cuda"
                
                logger.info("✓ Reranker loaded on GPU")
                self._check_gpu_memory("Reranker loaded")
            else:
                logger.warning("GPU not available, loading reranker on CPU...")
                
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                    reranker_model_path,
                    trust_remote_code=True
                )
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    reranker_model_path,
                    dtype=torch.float32,
                    trust_remote_code=True
                )
                self.reranker_model = self.reranker_model.to("cpu")
                self.reranker_device = "cpu"
                
                logger.info("✓ Reranker loaded on CPU")
            
            self.reranker_model.eval()
            logger.info(f"Jina reranker ready on {self.reranker_device}")
            
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            logger.warning("Will use retrieval without reranking")
            self.reranker_model = None
            self.reranker_tokenizer = None
            self.reranker_device = None
    
    def _init_llm(self):
        """Initialize Phi-4 LLM for answer generation - always on GPU with 4-bit quantization"""
        try:
            llm_model_path = os.getenv("LLM_MODEL", "microsoft/Phi-4-mini-instruct")
            
            logger.info(f"Loading Phi-4 LLM: {llm_model_path}")
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            
            # Always load LLM on GPU with 4-bit quantization for best performance
            if torch.cuda.is_available():
                logger.info("Loading LLM on GPU with 4-bit quantization...")
                
                # Clear cache before loading (Option 1: Memory Management)
                torch.cuda.empty_cache()
                self._check_gpu_memory("LLM loading")
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Use explicit device_map with reduced memory limit (Option 1)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_path,
                    quantization_config=quantization_config,
                    device_map={"": 0},  # Explicit GPU 0
                    dtype=torch.float16,
                    max_memory={0: "7.5GiB"},  # Reduced from 7GB - leaves room for router
                    low_cpu_mem_usage=True
                )
                self.llm_device = "cuda"
                logger.info("✓ LLM loaded on GPU with 4-bit quantization (NF4)")
                
                # Check memory after loading
                self._check_gpu_memory("LLM loaded")
            else:
                logger.warning("GPU not available, loading LLM on CPU...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_path,
                    dtype=torch.float32
                )
                self.llm_model = self.llm_model.to("cpu")
                self.llm_device = "cpu"
                logger.info("✓ LLM loaded on CPU")
            
            self.llm_model.eval()
            logger.info(f"Phi-4 LLM ready on {self.llm_device}")
        except Exception as e:
            logger.warning(f"Could not load Phi-4 LLM: {e}")
            logger.warning("Answer generation will not be available")
            self.llm_device = None
    
    def _init_keyllm(self):
        """Initialize KeyLLM for LLM-based keyword extraction using Qwen2.5-0.5B"""
        try:
            logger.info("Initializing KeyLLM with Qwen2.5-0.5B for keyword extraction...")
            
            # Load Qwen3-0.6B specifically for keyword extraction
            # Efficient model with good keyword extraction capabilities (~0.5 GB in 4-bit)
            keyword_model_name = "Qwen/Qwen3-0.6B"
            
            logger.info(f"Loading keyword extraction model: {keyword_model_name}")
            
            # Load tokenizer
            keyword_tokenizer = AutoTokenizer.from_pretrained(
                keyword_model_name,
                trust_remote_code=True
            )
            
            # Ensure pad_token is set
            if keyword_tokenizer.pad_token is None:
                keyword_tokenizer.pad_token = keyword_tokenizer.eos_token
            
            # Configure 4-bit quantization
            keyword_quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model with 4-bit quantization for efficiency
            keyword_model = AutoModelForCausalLM.from_pretrained(
                keyword_model_name,
                quantization_config=keyword_quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("✓ Qwen3-0.6B loaded for keyword extraction")
            
            # Create text generation pipeline for KeyBERT
            # Note: Don't pass device parameter when model is loaded with accelerate
            generator = pipeline(
                "text-generation",
                model=keyword_model,
                tokenizer=keyword_tokenizer,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.1,
                pad_token_id=keyword_tokenizer.eos_token_id
            )
            
            # Custom prompt for keyword extraction in Italian
            prompt = """
Ho il seguente documento:
[DOCUMENT]

Estrai le parole chiave più importanti da questo documento che sarebbero utili per la ricerca e il recupero di informazioni.
Restituisci solo le parole chiave separate da virgole, nient'altro.

Parole chiave:"""
            
            # Create TextGeneration instance for KeyLLM
            llm = TextGeneration(generator, prompt=prompt)
            self.keyllm_model = KeyLLM(llm)
            
            logger.info("✓ KeyLLM initialized with Qwen2.5-0.5B pipeline")
                
        except Exception as e:
            logger.warning(f"Could not initialize KeyLLM: {e}")
            logger.warning("Will fall back to traditional keyword extraction")
            self.keyllm_model = None
    
    def _check_gpu_memory(self, operation: str = "operation"):
        """Check available GPU memory before operations (Option 3: Memory Monitoring)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            available = total - reserved
            
            logger.memory(f"GPU Memory during {operation}:")
            logger.memory(f"  Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
            logger.memory(f"  Total: {total:.2f} GB | Available: {available:.2f} GB")
            
            if available < 0.5:  # Less than 500MB available
                logger.warning(f"⚠ Low GPU memory during {operation}: {available:.2f} GB available")
                torch.cuda.empty_cache()
                logger.info("  → Cleared GPU cache")
                # Check again after clearing
                allocated_new = torch.cuda.memory_allocated(0) / 1e9
                reserved_new = torch.cuda.memory_reserved(0) / 1e9
                available_new = total - reserved_new
                logger.memory(f"  → After clearing: {available_new:.2f} GB available")
    
    def process_document(self, file_path: Path, save_markdown: bool = True) -> Document:
        """
        Process a document using Docling and optionally save markdown
        
        Args:
            file_path: Path to document
            save_markdown: If True, save extracted markdown to processed folder
            
        Returns:
            LlamaIndex Document object
        """
        logger.info(f"Processing document with Docling: {file_path.name}")
        
        try:
            # Convert document using Docling
            conversion_result = self.doc_converter.convert(str(file_path))
            
            # Extract structured content
            doc_content = conversion_result.document
            
            # Get markdown representation (preserves structure)
            markdown_text = doc_content.export_to_markdown()
            
            # Save markdown to processed folder if requested
            if save_markdown:
                # Create category subfolder in processed directory
                category_folder = self.processed_path / file_path.parent.name
                category_folder.mkdir(parents=True, exist_ok=True)
                
                # Create markdown filename
                markdown_filename = file_path.stem + ".md"
                markdown_path = category_folder / markdown_filename
                
                # Save markdown file
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                
                logger.info(f"Saved markdown to {markdown_path}")
            
            # Create LlamaIndex Document with metadata
            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "category": file_path.parent.name,  # flood/wildfire
                "num_pages": len(doc_content.pages) if hasattr(doc_content, 'pages') else 0,
                "markdown_path": str(markdown_path) if save_markdown else None
            }
            
            document = Document(
                text=markdown_text,
                metadata=metadata,
                id_=str(file_path)
            )
            
            logger.info(f"Successfully processed {file_path.name} - {len(markdown_text)} characters")
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {file_path.name}: {e}")
            raise
    
    def load_markdown_document(self, markdown_path: Path) -> Document:
        """
        Load a document from a pre-processed markdown file
        
        Args:
            markdown_path: Path to markdown file
            
        Returns:
            LlamaIndex Document object
        """
        logger.info(f"Loading markdown document: {markdown_path.name}")
        
        try:
            # Read markdown content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            # Extract metadata from path structure
            category = markdown_path.parent.name
            original_filename = markdown_path.stem + ".pdf"
            
            metadata = {
                "filename": original_filename,
                "markdown_path": str(markdown_path),
                "file_size": markdown_path.stat().st_size,
                "category": category,
                "source": "cleaned_markdown"
            }
            
            document = Document(
                text=markdown_text,
                metadata=metadata,
                id_=str(markdown_path)
            )
            
            logger.info(f"Loaded markdown document - {len(markdown_text)} characters")
            return document
            
        except Exception as e:
            logger.error(f"Error loading markdown {markdown_path.name}: {e}")
            raise
    
    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Build vector index from documents using LlamaIndex
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            VectorStoreIndex
        """
        logger.info(f"Building index from {len(documents)} documents...")
        
        try:
            # Create index with custom embedding model and storage
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                transformations=[self.node_parser],
                show_progress=True
            )
            
            logger.info("Index built successfully")
            return self.index
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def query(self, query_text: str, top_k: int = None, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """
        Query the vector database using LlamaIndex with optional reranking
        
        Strategy: Retrieve 20 chunks with semantic search, then rerank to get top_k
        
        Args:
            query_text: Query string
            top_k: Number of final results to return (default: 5)
            use_reranker: Whether to use reranker model (default: True)
            
        Returns:
            List of relevant document chunks (reranked if available)
        """
        if top_k is None:
            top_k = int(os.getenv("TOP_K_RESULTS", 5))
        
        # Retrieve more candidates for reranking
        initial_k = 20  # Retrieve 20 candidates
        
        logger.debug(f"Querying: {query_text[:100]}...")
        logger.debug(f"Strategy: Retrieve {initial_k} → Rerank → Top {top_k}")
        
        try:
            # Load or create index if not exists
            if self.index is None:
                if self.chroma_collection.count() == 0:
                    logger.warning("No documents in collection. Please build knowledge base first.")
                    return []
                
                # Load existing index from vector store
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    embed_model=self.embed_model
                )
            
            # Create query engine with no LLM (retrieval only)
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=initial_k,  # Retrieve 20 candidates
            )
            
            # Execute query
            nodes = retriever.retrieve(query_text)
            
            # Format results
            formatted_results = []
            for node in nodes:
                formatted_results.append({
                    "content": node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, 'score') else None
                })
            
            logger.debug(f"Retrieved {len(formatted_results)} candidates")
            
            # Rerank if reranker is available and requested
            if use_reranker and self.reranker_model and len(formatted_results) > top_k:
                logger.debug(f"Reranking {len(formatted_results)} candidates...")
                formatted_results = self._rerank_results(query_text, formatted_results, top_k)
                logger.debug(f"After reranking: {len(formatted_results)} results")
            else:
                # Just return top_k without reranking
                formatted_results = formatted_results[:top_k]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank retrieved results using Jina reranker model
        
        Args:
            query: Query string
            results: List of retrieved results
            top_k: Number of top results to return
            
        Returns:
            Reranked and filtered results
        """
        if not self.reranker_model or not self.reranker_tokenizer:
            logger.warning("Reranker not available, returning original results")
            return results[:top_k]
        
        try:
            # Prepare pairs for reranking
            pairs = []
            for result in results:
                pairs.append([query, result['content']])
            
            # Tokenize all pairs
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.reranker_device)
                
                # Get relevance scores
                outputs = self.reranker_model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
            
            # Add reranker scores to results
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])
            
            # Sort by rerank score (higher is better)
            reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top_k
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            logger.warning("Falling back to original results")
            return results[:top_k]
    
    def _extract_keywords_with_llm(self, query: str, doc_samples: List[str], top_n: int = 10) -> set:
        """
        Extract keywords using KeyLLM with context from documents
        
        Args:
            query: User query
            doc_samples: Sample documents from knowledge base for context
            top_n: Number of keywords to extract (used for filtering results)
            
        Returns:
            Set of extracted keywords
        """
        if not self.keyllm_model:
            logger.warning("KeyLLM not available, falling back to regex")
            return set(re.findall(r'\b\w+\b', query.lower()))
        
        try:
            # Prepare document context (limit size)
            doc_context = "\n".join([doc[:200] for doc in doc_samples[:5]])
            
            # Combine query with document context
            # KeyLLM will use this combined text to extract relevant keywords
            combined_doc = f"{query}\n\n{doc_context}"
            
            # Extract keywords using KeyLLM
            # KeyLLM.extract_keywords() returns List[List[str]] for multiple docs
            # or List[str] for a single document
            logger.debug(f"Extracting keywords with KeyLLM...")
            keywords_result = self.keyllm_model.extract_keywords(
                docs=[combined_doc]
            )
            
            # Process results
            # KeyLLM returns List[List[str]] when given a list of docs
            # The LLM is expected to return comma-separated keywords
            keywords = set()
            
            if keywords_result and len(keywords_result) > 0:
                # Get the first document's keywords (List[str])
                keyword_list = keywords_result[0] if isinstance(keywords_result[0], list) else keywords_result
                
                # Process each keyword string
                for keyword_str in keyword_list[:top_n]:
                    if isinstance(keyword_str, str):
                        # Split by comma in case LLM returned comma-separated keywords
                        for keyword in keyword_str.split(','):
                            keyword = keyword.lower().strip()
                            # Filter out very short keywords and non-alphanumeric
                            if len(keyword) > 2 and any(c.isalnum() for c in keyword):
                                keywords.add(keyword)
                
                logger.debug(f"KeyLLM extracted {len(keywords)} keywords: {list(keywords)[:5]}...")
            
            if not keywords:
                logger.warning("KeyLLM returned no valid keywords, using fallback")
                return set(re.findall(r'\b\w+\b', query.lower()))
            
            return keywords
                
        except Exception as e:
            logger.error(f"Error extracting keywords with KeyLLM: {e}")
            logger.warning("Falling back to regex keyword extraction")
            import traceback
            traceback.print_exc()
            return set(re.findall(r'\b\w+\b', query.lower()))
    
    def keyword_search(self, query_text: str, top_k: int = None, use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search on document chunks using LLM-enhanced keyword extraction
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            use_llm: If True, use KeyLLM for keyword extraction (default: True)
            
        Returns:
            List of relevant document chunks with keyword match scores
        """
        if top_k is None:
            top_k = int(os.getenv("TOP_K_RESULTS", 5))
        
        logger.debug(f"Keyword search: {query_text[:100]}...")
        
        try:
            # Load index if needed
            if self.index is None:
                if self.chroma_collection.count() == 0:
                    logger.warning("No documents in collection. Please build knowledge base first.")
                    return []
                
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    embed_model=self.embed_model
                )
            
            # Get all documents from the collection
            all_docs = self.chroma_collection.get()
            
            if not all_docs or 'documents' not in all_docs:
                return []
            
            # Extract keywords using LLM or traditional method
            if use_llm and self.keyllm_model:
                logger.debug("Using KeyLLM for keyword extraction...")
                query_keywords = self._extract_keywords_with_llm(
                    query_text, 
                    all_docs['documents'][:10],  # Sample first 10 docs for context
                    top_n=10
                )
            else:
                # Fallback to simple regex extraction
                logger.debug("Using traditional keyword extraction...")
                query_keywords = set(re.findall(r'\b\w+\b', query_text.lower()))
            
            logger.debug(f"Extracted keywords: {query_keywords}")
            
            # Score documents based on keyword matches
            scored_docs = []
            for i, doc_text in enumerate(all_docs['documents']):
                doc_text_lower = doc_text.lower()
                
                # Count keyword matches
                match_count = sum(1 for keyword in query_keywords if keyword in doc_text_lower)
                
                if match_count > 0:
                    # Calculate score based on match count and document length
                    score = match_count / len(query_keywords) if query_keywords else 0
                    
                    scored_docs.append({
                        'content': doc_text,
                        'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {},
                        'score': score,
                        'matches': match_count,
                        'keywords': list(query_keywords)[:10]  # Store keywords used
                    })
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            
            logger.debug(f"Found {len(scored_docs[:top_k])} keyword matches")
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query_text: str, top_k: int = None, 
                     semantic_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining semantic and keyword search
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1), keyword weight = 1 - semantic_weight
            
        Returns:
            List of relevant document chunks with combined scores
        """
        if top_k is None:
            top_k = int(os.getenv("TOP_K_RESULTS", 5))
        
        logger.info(f"Hybrid search: {query_text[:100]}...")
        logger.info(f"Weights: semantic={semantic_weight}, keyword={1-semantic_weight}")
        
        try:
            # Get results from both methods (fetch more to have options for reranking)
            fetch_k = top_k * 3
            semantic_results = self.query(query_text, top_k=fetch_k)
            keyword_results = self.keyword_search(query_text, top_k=fetch_k)
            
            # Normalize scores to 0-1 range
            if semantic_results and semantic_results[0].get('score'):
                max_sem_score = max(r['score'] for r in semantic_results if r.get('score'))
                for r in semantic_results:
                    if r.get('score'):
                        r['norm_score'] = r['score'] / max_sem_score if max_sem_score > 0 else 0
                    else:
                        r['norm_score'] = 0
            
            if keyword_results:
                max_kw_score = max(r['score'] for r in keyword_results if r.get('score'))
                for r in keyword_results:
                    if r.get('score'):
                        r['norm_score'] = r['score'] / max_kw_score if max_kw_score > 0 else 0
                    else:
                        r['norm_score'] = 0
            
            # Combine results by content
            combined = {}
            
            # Add semantic results
            for result in semantic_results:
                content_key = result['content'][:100]  # Use first 100 chars as key
                combined[content_key] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'semantic_score': result.get('norm_score', 0),
                    'keyword_score': 0
                }
            
            # Add/merge keyword results
            for result in keyword_results:
                content_key = result['content'][:100]
                if content_key in combined:
                    combined[content_key]['keyword_score'] = result.get('norm_score', 0)
                else:
                    combined[content_key] = {
                        'content': result['content'],
                        'metadata': result['metadata'],
                        'semantic_score': 0,
                        'keyword_score': result.get('norm_score', 0)
                    }
            
            # Calculate hybrid scores
            hybrid_results = []
            for item in combined.values():
                hybrid_score = (
                    semantic_weight * item['semantic_score'] + 
                    (1 - semantic_weight) * item['keyword_score']
                )
                hybrid_results.append({
                    'content': item['content'],
                    'metadata': item['metadata'],
                    'score': hybrid_score,
                    'semantic_score': item['semantic_score'],
                    'keyword_score': item['keyword_score']
                })
            
            # Sort by hybrid score and return top_k
            hybrid_results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Hybrid search returned {len(hybrid_results[:top_k])} results")
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], 
                       max_new_tokens: int = 512) -> str:
        """
        Generate answer using Phi-4 LLM with retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved document chunks
            max_new_tokens: Maximum tokens to generate (default: 512 for complete answers)
            
        Returns:
            Generated answer
        """
        if not self.llm_model or not self.llm_tokenizer:
            logger.warning("LLM not available. Cannot generate answer.")
            return "LLM not available for answer generation."
        
        try:
            # Prepare context from chunks
            context_text = "\n\n".join([
                f"[{i+1}] {chunk['content'][:500]}..."  # Limit context length
                for i, chunk in enumerate(context_chunks[:5])  # Use top 5 chunks
            ])
            
            # Create chat prompt - allow full answers but encourage conciseness
            chat = [
                {
                    "role": "system",
                    "content": "Sei un assistente esperto in analisi di documenti ISPRA. Fornisci risposte complete e accurate, ma mantienile concise evitando ripetizioni."
                },
                {
                    "role": "user",
                    "content": f"""Basandoti sui seguenti estratti di documenti ISPRA, rispondi alla domanda in modo completo.

Contesto:
{context_text}

Domanda: {query}

IMPORTANTE: La tua risposta deve essere:
- COMPLETA: fornisci tutti i dettagli necessari per rispondere pienamente
- CONCISA: sii breve ma esaustivo, evita ripetizioni e ridondanze
- ACCURATA: cita solo i dati specifici dal contesto fornito
- DIRETTA: vai subito al punto senza introduzioni inutili

Risposta:"""
                }
            ]
            
            # Apply chat template
            chat_formatted = self.llm_tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            input_tokens = self.llm_tokenizer(
                chat_formatted, 
                return_tensors="pt"
            ).to(self.llm_device)
            
            # Generate
            logger.info(f"Generating answer with Phi-4 LLM...")
            
            # Get the length of the input to only decode new tokens
            input_length = input_tokens['input_ids'].shape[1]
            
            output = self.llm_model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id
            )
            
            # Decode only the generated tokens (skip the input prompt)
            generated_tokens = output[0][input_length:]
            answer = self.llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            logger.info(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query_and_answer(self, query: str, top_k: int = 5, 
                        use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate answer
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            use_hybrid: Use hybrid retrieval (True) or semantic only (False)
            
        Returns:
            Dictionary with answer, retrieved chunks, and metadata
        """
        logger.info(f"Query and Answer: {query}")
        
        # Retrieve relevant chunks
        if use_hybrid:
            chunks = self.hybrid_search(query, top_k=top_k)
        else:
            chunks = self.query(query, top_k=top_k)
        
        if not chunks:
            return {
                'query': query,
                'answer': "Non ho trovato documenti rilevanti per rispondere alla domanda.",
                'chunks': [],
                'method': 'hybrid' if use_hybrid else 'semantic'
            }
        
        # Generate answer
        answer = self.generate_answer(query, chunks)
        
        return {
            'query': query,
            'answer': answer,
            'chunks': chunks,
            'method': 'hybrid' if use_hybrid else 'semantic',
            'num_chunks': len(chunks)
        }
    
    def build_knowledge_base_from_markdown(self, force_rebuild: bool = False) -> None:
        """
        Build knowledge base from manually cleaned markdown files
        
        This is the main method for building the knowledge base.
        Workflow:
        1. Run extract_markdown.py to extract markdown from PDFs
        2. Manually clean markdown files in data/processed/
        3. Run this method to chunk and encode to vector database
        
        Args:
            force_rebuild: If True, rebuild even if database exists
        """
        logger.info("Building knowledge base from cleaned markdown files...")
        
        # Check if already built
        if not force_rebuild and self.chroma_collection.count() > 0:
            logger.info(f"Knowledge base already contains {self.chroma_collection.count()} nodes")
            logger.info("Use force_rebuild=True to rebuild from scratch")
            # Load existing index
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
            return
        
        # Find all markdown files (recursively search subdirectories)
        markdown_files = list(self.processed_path.rglob("*.md"))
        
        if not markdown_files:
            logger.error(f"No markdown files found in {self.processed_path}")
            logger.info("Please run: python scripts/extract_markdown.py")
            logger.info("Then manually clean the markdown files before building the knowledge base")
            return
        
        logger.info(f"Found {len(markdown_files)} markdown files to process")
        
        # Load each markdown document
        documents = []
        for md_file in markdown_files:
            try:
                doc = self.load_markdown_document(md_file)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {md_file.name}: {e}")
        
        # Build index with LlamaIndex
        if documents:
            self.build_index(documents)
            logger.info(f"Knowledge base built with {len(documents)} cleaned markdown documents")
        else:
            logger.warning("No markdown documents loaded successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_chunks": self.chroma_collection.count(),
            "collection_name": self.chroma_collection.name,
            "db_path": str(self.db_path)
        }
    
    def calculate_similarity(self, queries: List[str], passages: List[str]) -> Any:
        """
        Calculate cosine similarity between queries and passages
        
        Args:
            queries: List of query strings
            passages: List of passage strings
            
        Returns:
            Cosine similarity matrix
        """
        logger.info(f"Calculating similarity for {len(queries)} queries and {len(passages)} passages")
        
        # Encode queries and passages using the embed model
        # HuggingFaceEmbedding has get_text_embedding method
        if hasattr(self.sentence_transformer, 'encode'):
            # It's a SentenceTransformer
            query_embeddings = self.sentence_transformer.encode(queries)
            passage_embeddings = self.sentence_transformer.encode(passages)
        else:
            # It's HuggingFaceEmbedding, use get_text_embedding
            query_embeddings = [self.embed_model.get_text_embedding(q) for q in queries]
            passage_embeddings = [self.embed_model.get_text_embedding(p) for p in passages]
        
        # Calculate cosine similarity
        similarities = util.cos_sim(query_embeddings, passage_embeddings)
        
        return similarities
