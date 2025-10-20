"""
Core agent logic implementing streamlined RAG workflow with Qwen3-1.7B router

Workflow: Input → Router (Qwen3-1.7B) → RAG/TOOL/DIRECT → Final Answer

The agent:
1. Routes the query using Qwen3-1.7B with few-shot learning
2. Determines one of three paths: RAG, TOOL, or DIRECT
3. Processes based on routing decision
4. Uses main LLM to generate the final answer
"""

import json
import logging
import re
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rag_pipeline import RAGPipeline
from tool_caller import ToolCaller
from utils import load_config

logger = logging.getLogger(__name__)

LANGCHAIN_AVAILABLE = False


# Italian city coordinates for spatial queries
ITALIAN_CITY_COORDINATES = {
    'roma': (41.9028, 12.4964),
    'milano': (45.4642, 9.1900),
    'napoli': (40.8518, 14.2681),
    'torino': (45.0703, 7.6869),
    'palermo': (38.1157, 13.3615),
    'bologna': (44.4949, 11.3426),
    'firenze': (43.7696, 11.2558),
    'venezia': (45.4408, 12.3155),
    'genova': (44.4056, 8.9463),
    'bari': (41.1171, 16.8719),
    'catania': (37.5079, 15.0830),
    'messina': (38.1937, 15.5542),
    'verona': (45.4384, 10.9916),
    'padova': (45.4064, 11.8768),
    'trieste': (45.6495, 13.7768),
    'brescia': (45.5416, 10.2118),
    'parma': (44.8015, 10.3279),
    'modena': (44.6471, 10.9252),
    'reggio emilia': (44.6989, 10.6297),
    'pisa': (43.7228, 10.4017),
    'livorno': (43.5485, 10.3106),
    'cagliari': (39.2238, 9.1217),
    'perugia': (43.1107, 12.3908),
    'ancona': (43.6158, 13.5189),
    'pescara': (42.4584, 14.2139),
    'l\'aquila': (42.3498, 13.3995),
    'campobasso': (41.5630, 14.6631),
    'potenza': (40.6389, 15.8056),
    'cosenza': (39.2979, 16.2542),
    'catanzaro': (38.9097, 16.5877),
    'reggio calabria': (38.1113, 15.6476),
    'lecce': (40.3515, 18.1750),
    'foggia': (41.4621, 15.5447),
    'taranto': (40.4761, 17.2305),
}

# Risk color to class mapping for flood alerts
RISK_COLOR_MAPPING = {
    'verde': 'Assente',
    'gialla': 'Ordinaria',
    'giallo': 'Ordinaria',
    'arancione': 'Moderata',
    'arancio': 'Moderata',
    'rossa': 'Elevata',
    'rosso': 'Elevata',
}


class QueryType(Enum):
    """Query classification types"""
    RAG = "RAG"           # Historical data from documents/reports
    TOOL = "TOOL"         # Recent/real-time data access
    DIRECT = "DIRECT"     # Domain-related questions (floods/wildfires)
    NONE = "NONE"         # Out of scope / guardrail


# Few-shot examples for router prompt (all in Italian)
ROUTER_PROMPT_TEMPLATE = """Classifica questa query in una categoria: RAG, TOOL, DIRECT o NONE.

REGOLE:
- RAG = dati storici da documenti (anni 2018, 2021, 2022, 2023, 2024, report ISPRA)
- TOOL = dati recenti in tempo reale (oggi, adesso, ultime ore, in questo momento)  
- DIRECT = domande generali (cos'è, spiega, come funziona, cause, prevenzione)
- NONE = altri argomenti non correlati ad alluvioni/incendi

ESEMPI:

Query: "Quanta superficie è stata bruciata nel 2021?"
Output: RAG

Query: "Quale regione più colpita nel 2021?"
Output: RAG

Query: "Comuni a rischio alluvioni nel 2018?"
Output: RAG

Query: "Incendi in Sicilia oggi?"
Output: TOOL

Query: "Allerte in Emilia-Romagna adesso?"
Output: TOOL

Query: "Cos'è un'alluvione?"
Output: DIRECT

Query: "Come prevenire gli incendi?"
Output: DIRECT

Query: "Ciao come stai?"
Output: NONE

Query: "Ricetta pizza margherita"
Output: NONE

Query: "{query}"
Output:"""


class Agent:
    """
    Streamlined agent with Qwen3-1.7B router
    
    Workflow:
    1. Input → Router (Qwen3-1.7B with few-shot learning)
    2. Router classifies: RAG, TOOL, or DIRECT
    3. Process based on classification
    4. Generate final answer with main LLM
    """
    
    def __init__(self):
        """Initialize agent components with all models on GPU"""
        logger.info("Initializing Agent with Qwen3 models (all on GPU)...")
        
        # Load configuration
        self.config = load_config()
        
        # Note: Document context loading removed - using simplified prompt
        
        # Load Router model (Qwen3-1.7B)
        logger.info("Step 1/3: Loading Router model (Qwen3-1.7B)...")
        self._init_router()
        
        # Clear cache before loading RAG pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Step 2/3: Cleared GPU cache before loading RAG pipeline")
        
        # Initialize RAG pipeline (includes main LLM, embedding, reranker, keyword LLM)
        logger.info("Step 3/3: Loading RAG pipeline...")
        self.rag_pipeline = RAGPipeline()
        
        # Initialize tool caller (for data file access)
        self.tool_caller = ToolCaller()
        
        # Initialize LangChain agent if available
        self.langchain_agent = None
        if LANGCHAIN_AVAILABLE:
            try:
                logger.info("Step 4/4 (Optional): Initializing LangChain agent...")
                # Use minimal settings to avoid API requirements
                self.langchain_agent = None  # We'll initialize on-demand
                logger.info("✓ LangChain agent ready (will initialize on first use)")
            except Exception as e:
                logger.warning(f"Could not initialize LangChain agent: {e}")
                self.langchain_agent = None
        
        # Agent state
        self.conversation_history = []
        
        logger.info("✓ Agent initialized successfully")
        logger.info("✓ All models loaded on GPU:")
        logger.info("  - Router: Qwen3-1.7B (1.26 GB)")
        logger.info("  - Main LLM: Phi-4-mini-instruct (2.69 GB)")
        logger.info("  - Keyword LLM: Qwen3-0.6B (0.51 GB)")
        logger.info("  - Embedding: Granite-278M (1.04 GB)")
        logger.info("  - Reranker: Jina-v2-base (0.52 GB)")
        logger.info("  - Total: ~6 GB / 8 GB (75% GPU usage)")
    
    def _load_document_context(self) -> str:
        """
        Load all .md files from data/processed and create a summary for the router
        
        Returns:
            String with document summaries
        """
        try:
            processed_dir = Path("data/processed")
            md_files = list(processed_dir.glob("**/*.md"))
            
            if not md_files:
                logger.warning("No .md files found in data/processed")
                return "Nessun documento disponibile."
            
            context_parts = []
            context_parts.append(f"📚 DOCUMENTI DISPONIBILI ({len(md_files)} files con dati storici):\n")
            
            for md_file in md_files:
                # Read more content of each file for better context
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # Increased from 500 to 1000
                        # Extract filename and relative path
                        rel_path = md_file.relative_to(processed_dir)
                        
                        # Create a more informative summary
                        file_summary = f"\n📄 {rel_path}:\n"
                        
                        # Extract year from filename if present
                        import re
                        year_match = re.search(r'20\d{2}', str(md_file))
                        if year_match:
                            file_summary += f"   Anno: {year_match.group()}\n"
                        
                        # Add first 400 chars of content
                        file_summary += f"   Contenuto: {content[:400]}...\n"
                        
                        context_parts.append(file_summary)
                except Exception as e:
                    logger.warning(f"Could not read {md_file}: {e}")
            
            context = "\n".join(context_parts)
            logger.info(f"✓ Loaded context from {len(md_files)} documents (~{len(context)} chars)")
            return context
            
        except Exception as e:
            logger.error(f"Error loading document context: {e}")
            return "Errore nel caricamento dei documenti."
    
    def _check_gpu_memory(self, operation: str = "operation"):
        """Check available GPU memory (Option 3: Memory Monitoring)"""
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
    
    def _init_router(self):
        """Initialize Qwen3-1.7B as router on GPU with 4-bit quantization"""
        logger.info("Initializing Qwen3-1.7B router with thinking capabilities...")
        
        try:
            router_model_path = "Qwen/Qwen3-1.7B"
            
            # Always load router on GPU with 4-bit quantization for best performance
            if torch.cuda.is_available():
                logger.info("Loading Qwen3-1.7B router on GPU with 4-bit quantization...")
                
                # Clear cache and check memory before loading
                torch.cuda.empty_cache()
                self._check_gpu_memory("Router loading")
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.router_tokenizer = AutoTokenizer.from_pretrained(
                    router_model_path,
                    trust_remote_code=True
                )
                
                # Load Qwen3-1.7B with 4-bit quantization
                self.router_model = AutoModelForCausalLM.from_pretrained(
                    router_model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.router_device = "cuda"
                
                logger.info("✓ Qwen3-1.7B Router loaded on GPU with 4-bit quantization")
                
                # Check memory after loading
                self._check_gpu_memory("Router loaded")
            else:
                logger.warning("GPU not available, loading router on CPU...")
                
                self.router_tokenizer = AutoTokenizer.from_pretrained(
                    router_model_path,
                    trust_remote_code=True
                )
                
                self.router_model = AutoModelForCausalLM.from_pretrained(
                    router_model_path,
                    dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.router_model = self.router_model.to("cpu")
                self.router_device = "cpu"
                
                logger.info("✓ Router loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            traceback.print_exc()
            # Fallback to keyword-based routing
            self.router_model = None
            self.router_tokenizer = None
            self.router_device = None
            logger.warning("Router initialization failed, will use keyword-based routing")
    
    def route(self, query: str, enable_thinking: bool = False) -> QueryType:
        """
        Route query using Qwen3-1.7B with optional reasoning capabilities
        
        Uses Qwen3's built-in thinking mode to provide transparent reasoning 
        before classification when enabled.
        
        Args:
            query: User query
            enable_thinking: Whether to enable Qwen3's reasoning mode (default: False)
            
        Returns:
            QueryType indicating processing path
        """
        logger.debug(f"Routing query: {query[:100]}... (thinking: {enable_thinking})")
        
        # If router model failed to load, use fallback
        if self.router_model is None:
            return self._fallback_route(query)
        
        try:
            # Build routing prompt with few-shot examples
            routing_prompt = ROUTER_PROMPT_TEMPLATE.format(query=query)
            
            # Use different approaches based on thinking mode
            if enable_thinking:
                # Use Qwen3's chat template with thinking mode for reasoning
                messages = [
                    {"role": "user", "content": routing_prompt}
                ]
                text = self.router_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                model_inputs = self.router_tokenizer([text], return_tensors="pt").to(self.router_device)
                max_tokens = 512
            else:
                # Use direct prompting (original approach) for fast classification
                model_inputs = self.router_tokenizer(
                    routing_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                ).to(self.router_device)
                max_tokens = 10
            
            # Generate classification
            with torch.no_grad():
                generated_ids = self.router_model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.router_tokenizer.eos_token_id
                )
            
            # Extract output tokens (excluding input)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            thinking_content = ""
            content = ""
            
            # Parse thinking content only if thinking mode is enabled
            if enable_thinking:
                # Parse thinking content (Qwen3 uses token 151668 for </think>)
                try:
                    # Find the </think> token to separate reasoning from final answer
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    # No thinking delimiter found, treat all as content
                    index = 0
                
                # Decode thinking and content separately
                thinking_content = self.router_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.router_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                
                # Log the reasoning process
                if thinking_content:
                    logger.info(f"→ Router reasoning: {thinking_content[:200]}...")
            else:
                # No thinking mode, decode all as content
                content = self.router_tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            # Extract classification from content
            classification = self._parse_router_output(content)
            
            # Log the actual router output for debugging
            logger.info(f"→ Router raw output: '{content[:200]}'")
            logger.info(f"→ Router decision: {classification.value}")
            return classification
            
        except Exception as e:
            logger.error(f"Router error: {e}, falling back to keyword-based routing")
            return self._fallback_route(query)
    
    def _parse_router_output(self, output: str) -> QueryType:
        """
        Parse router output to extract classification
        
        Args:
            output: Router model output
            
        Returns:
            QueryType classification
        """
        output_upper = output.upper().strip()
        
        # Look for explicit "Output: CATEGORY" pattern first (from few-shot examples)
        import re
        output_match = re.search(r'OUTPUT:\s*(TOOL|RAG|DIRECT|NONE)', output_upper)
        if output_match:
            category = output_match.group(1)
            if category == "TOOL":
                return QueryType.TOOL
            elif category == "RAG":
                return QueryType.RAG
            elif category == "DIRECT":
                return QueryType.DIRECT
            elif category == "NONE":
                return QueryType.NONE
        
        # Fallback: Look for the classification anywhere in the output
        # Check order matters - check most specific first and look for standalone words
        if re.search(r'\bTOOL\b', output_upper):
            return QueryType.TOOL
        elif re.search(r'\bRAG\b', output_upper):
            return QueryType.RAG
        elif re.search(r'\bDIRECT\b', output_upper):
            return QueryType.DIRECT
        elif re.search(r'\bNONE\b', output_upper):
            return QueryType.NONE
        else:
            # Default to NONE as guardrail
            logger.warning(f"Unclear router output: '{output}', defaulting to NONE")
            return QueryType.NONE
    
    def _fallback_route(self, query: str) -> QueryType:
        """
        Fallback keyword-based routing if router model unavailable
        
        Args:
            query: User query
            
        Returns:
            QueryType indicating processing path
        """
        logger.debug("Using fallback keyword-based routing")
        
        query_lower = query.lower()
        
        # Keywords for TOOL (recent/real-time data)
        tool_keywords = [
            "ultime ore", "ultimi giorni", "ultime 24 ore", "oggi", "adesso",
            "in questo momento", "ora", "corrente", "attuale", "recente",
            "right now", "now", "today", "current", "latest hours"
        ]
        
        # Keywords for RAG (historical data from reports)
        rag_keywords = [
            "2021", "2022", "2023", "2024", "2020", "anni", "storico",
            "rapporto", "report", "ispra", "documento", "secondo",
            "negli ultimi anni", "nel", "erano", "passato"
        ]
        
        # Keywords for DIRECT (general domain questions)
        direct_keywords = [
            "cos'è", "cosa sono", "come si", "quali sono", "spiega",
            "definizione", "cause", "prevenzione", "rischi", "spiegazione",
            "what is", "explain", "how to", "causes", "prevention"
        ]
        
        # Domain keywords (floods/wildfires)
        domain_keywords = [
            "incendi", "incendio", "alluvioni", "alluvione", "dissesto",
            "idrogeologico", "hotspot", "fuoco", "boschiv", "forestale",
            "inondazione", "fire", "flood", "wildfire"
        ]
        
        # Check if query is about floods/wildfires
        is_domain_related = any(kw in query_lower for kw in domain_keywords)
        
        if not is_domain_related:
            # Not about floods/wildfires -> NONE
            logger.info("→ Fallback route: NONE (out of scope)")
            return QueryType.NONE
        
        # Domain related - now classify the type
        if any(kw in query_lower for kw in tool_keywords):
            logger.info("→ Fallback route: TOOL (recent data)")
            return QueryType.TOOL
        
        if any(kw in query_lower for kw in rag_keywords):
            logger.info("→ Fallback route: RAG (historical data)")
            return QueryType.RAG
        
        if any(kw in query_lower for kw in direct_keywords):
            logger.info("→ Fallback route: DIRECT (general question)")
            return QueryType.DIRECT
        
        # Domain related but unclear -> default to NONE for safety
        logger.info("→ Fallback route: NONE (default - unclear intent)")
        return QueryType.NONE
    
    def process(self, query: str, use_hybrid: bool = True, top_k: int = 5, enable_thinking: bool = False) -> Dict[str, Any]:
        """
        Main processing method implementing the streamlined workflow
        
        Workflow:
        1. Input → Router (Qwen3-1.7B)
        2. Router decides: TOOL, RAG, DIRECT, or NONE
        3. Process based on routing decision
        4. Generate final answer with main LLM
        
        Args:
            query: User query
            use_hybrid: Use hybrid retrieval (semantic + keyword)
            top_k: Number of chunks to retrieve if RAG is needed
            enable_thinking: Enable router's reasoning mode (default: False)
            
        Returns:
            Dictionary with answer, metadata, and processing info
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        try:
            # STEP 1: Route query with Qwen3-1.7B
            query_type = self.route(query, enable_thinking=enable_thinking)
            
            # STEP 2: Process based on routing decision
            if query_type == QueryType.NONE:
                # Out of scope - return guardrail message
                result = {
                    'query': query,
                    'answer': "Mi dispiace, posso rispondere solo a domande riguardanti alluvioni e incendi in Italia. Come posso aiutarti su questi argomenti?",
                    'method': 'none',
                    'num_chunks': 0
                }
            
            elif query_type == QueryType.TOOL:
                result = self._process_with_tools(query)
            
            elif query_type == QueryType.RAG:
                # Use RAG pipeline
                result = self.rag_pipeline.query_and_answer(
                    query=query,
                    top_k=top_k,
                    use_hybrid=use_hybrid
                )
            
            else:  # DIRECT
                # Generate answer without context
                result = self._generate_direct_answer(query)
            
            # Add metadata
            result['query_type'] = query_type.value
            result['timestamp'] = self._get_timestamp()
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": result['answer']
            })
            
            logger.info(f"✓ Answer generated successfully ({query_type.value})")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = {
                'query': query,
                'answer': f"Mi dispiace, si è verificato un errore: {str(e)}",
                'error': str(e),
                'query_type': 'error'
            }
            return error_response
    
    def _process_with_tools(self, query: str) -> Dict[str, Any]:
        """
        Process query requiring data file access with FORCED tool calling.
        
        This method implements a specialized agent that:
        1. MUST select an appropriate tool (no option to skip)
        2. Extracts parameters from the query
        3. Validates parameters using Pydantic schemas
        4. Executes the tool and returns structured output
        5. Uses tool output directly as the answer
        
        Args:
            query: User query
            
        Returns:
            Result dictionary with answer, tool_name, tool_input, and tool_output
        """
        logger.info("Processing with FORCED tool calling agent...")
        
        try:
            # Step 1: Get all available tools with descriptions
            if LANGCHAIN_AVAILABLE:
                try:
                    from langchain_tools import get_tool_descriptions
                    tool_descriptions = get_tool_descriptions()
                    logger.info(f"Loaded {len(tool_descriptions)} tool descriptions from LangChain")
                except Exception as e:
                    logger.warning(f"Could not load LangChain tools: {e}")
                    tool_descriptions = self._get_fallback_tool_descriptions()
            else:
                tool_descriptions = self._get_fallback_tool_descriptions()
            
            # Step 2: Force tool selection using Phi-4
            tool_name, tool_params = self._force_tool_selection(query, tool_descriptions)
            
            logger.info(f"Selected tool: {tool_name}")
            logger.info(f"Extracted parameters: {tool_params}")
            
            # Step 3: Execute the selected tool
            tool_result = self.tool_caller.execute(tool_name, tool_params)
            
            # Step 4: Use tool output directly as answer (optionally format with Phi-4)
            if self.rag_pipeline.llm_model:
                # Format the tool result into a natural language answer
                answer = self._format_tool_result(query, tool_name, tool_params, tool_result)
            else:
                # Direct output if no LLM available
                answer = self._format_tool_result_simple(tool_result)
            
            # Step 5: Return structured output
            return {
                'query': query,
                'answer': answer,
                'tool_name': tool_name,
                'tool_input': tool_params,
                'tool_output': tool_result,
                'method': 'forced_tool_calling'
            }
            
        except Exception as e:
            logger.error(f"Error in forced tool calling: {e}")
            return {
                'query': query,
                'answer': f"Mi dispiace, si è verificato un errore nell'esecuzione del tool: {str(e)}",
                'error': str(e),
                'method': 'forced_tool_calling_error'
            }
    
    def _get_fallback_tool_descriptions(self) -> Dict[str, str]:
        """
        Get fallback tool descriptions if LangChain not available.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {
            # Fire Hotspots - Geographic
            'get_hotspots_by_region': 'Ottieni tutti i punti caldi di incendio in una specifica regione italiana. Parametri: region_name (str)',
            'get_hotspots_by_province': 'Ottieni tutti i punti caldi di incendio in una specifica provincia italiana. Parametri: province_name (str)',
            'get_hotspots_by_municipality': 'Ottieni tutti i punti caldi di incendio in un specifico comune italiano. Parametri: municipality_name (str)',
            
            # Fire Hotspots - Attributes
            'get_hotspots_by_intensity': 'Filtra punti caldi per livello di intensità (low/medium/high). Parametri: intensity (str)',
            'get_hotspots_by_confidence': 'Filtra punti caldi per livello di confidenza (nominal/low/high). Parametri: confidence (str)',
            'get_hotspots_by_sensor': 'Filtra punti caldi per tipo di sensore (MODIS/VIIRS). Parametri: sensor (str)',
            'get_hotspots_by_satellite': 'Filtra punti caldi per satellite (Terra/Aqua/NPP/NOAA-20/NOAA-21). Parametri: satellite (str)',
            
            # Fire Hotspots - Temporal
            'get_hotspots_by_date': 'Filtra punti caldi per data specifica. Parametri: date (str formato YYYY-MM-DD)',
            'get_hotspots_by_time_of_day': 'Filtra punti caldi per periodo del giorno (day/night). Parametri: time_period (str)',
            'get_hotspot_statistics': 'Ottieni statistiche complete sui punti caldi di incendio. Nessun parametro richiesto.',
            
            # Fire Hotspots - Spatial
            'get_hotspots_within_distance': 'Trova hotspots entro una distanza da coordinate. Parametri: latitude, longitude, distance_km',
            'get_hotspots_in_bounding_box': 'Trova hotspots in un riquadro geografico. Parametri: min_lat, max_lat, min_lon, max_lon',
            
            # Flood Bulletins - Geographic
            'get_flood_zones_by_region': 'Ottieni zone a rischio alluvione per regione. Parametri: region_name (str)',
            'get_flood_zones_by_risk_level': 'Filtra zone per livello di rischio (oggi/domani). Parametri: risk_level (str), date_type (str)',
            'get_flood_zones_by_risk_class': 'Filtra zone per classe di rischio (Assente/Ordinaria/Moderata/Elevata). Parametri: risk_class (str)',
            'get_flood_zones_by_minimum_risk_class': 'Filtra zone con rischio minimo. Parametri: minimum_risk_class (str)',
            
            # Flood Bulletins - Search
            'get_flood_zones_by_zone_code': 'Cerca zona per codice. Parametri: zone_code (str)',
            'get_flood_zones_with_name_pattern': 'Cerca zone per pattern nel nome. Parametri: name_pattern (str)',
            
            # Flood Bulletins - Spatial
            'get_flood_zones_within_distance': 'Trova zone entro distanza da coordinate. Parametri: latitude, longitude, distance_km',
            'get_flood_zones_in_bounding_box': 'Trova zone in riquadro geografico. Parametri: min_lat, max_lat, min_lon, max_lon',
            'get_flood_statistics': 'Ottieni statistiche complete sulle alluvioni. Nessun parametro richiesto.',
            'compare_flood_bulletins': 'Confronta bollettini alluvioni oggi vs domani. Nessun parametro richiesto.',
            
            # Utilities
            'get_available_regions': 'Ottieni lista di tutte le regioni disponibili. Nessun parametro richiesto.',
            'get_data_summary': 'Ottieni riepilogo completo dei dati disponibili. Nessun parametro richiesto.',
            'get_vigilance_data_summary': 'Ottieni riepilogo completo dei dati di vigilanza. Nessun parametro richiesto.',
            'list_available_files': 'Elenca tutti i file di dati disponibili. Nessun parametro richiesto.'
        }
    
    def _force_tool_selection(self, query: str, tool_descriptions: Dict[str, str]) -> tuple:
        """
        Force selection of appropriate tool using Phi-4 with all tool descriptions.
        
        Args:
            query: User query
            tool_descriptions: Dictionary of tool names to descriptions
            
        Returns:
            Tuple of (tool_name, parameters)
        """
        # Build prompt with ALL tool descriptions
        tools_text = "\n".join([
            f"- {name}: {desc}"
            for name, desc in tool_descriptions.items()
        ])
        
        prompt = f"""Sei un assistente specializzato nella selezione di tool per rispondere a domande su incendi e alluvioni in Italia.

TOOL DISPONIBILI:
{tools_text}

ISTRUZIONI OBBLIGATORIE:
1. DEVI selezionare ESATTAMENTE UN tool dalla lista sopra
2. DEVI estrarre i parametri necessari dalla domanda dell'utente
3. NON puoi rispondere senza selezionare un tool
4. Se la domanda riguarda incendi, usa i tool get_hotspots_*
5. Se la domanda riguarda alluvioni, usa i tool get_flood_zones_*
6. Se la domanda chiede statistiche generali, usa get_hotspot_statistics o get_flood_statistics
7. Se la domanda chiede regioni disponibili, usa get_available_regions
8. Se la domanda chiede un riepilogo generale, usa get_data_summary

REGOLE PARAMETRI OBBLIGATORIE:
- Per intensità hotspot usa: "Bassa", "Media", "Alta", "Molto Alta", "Sconosciuta" (param: intensity_level)
- Per confidenza hotspot usa: "Alta", "Media", "Bassa", "Sconosciuta" (param: confidence_category)
- Per sensore usa: "MODIS" o "VIIRS" (param: sensor_type)
- Per satellite usa: "Terra", "Aqua", "N", "J1", "J2" (param: satellite_name)
- Per periodo giorno usa: "Giorno" o "Notte" (param: time_period)
- Per livello rischio alluvioni usa: "Verde", "Giallo", "Arancione", "Rosso" (param: risk_level)
- Per classe rischio usa numeri: 0 (Verde), 1 (Giallo), 2 (Arancione), 3 (Rosso) (param: risk_class o min_risk_class)
- Per tipo bollettino usa: "oggi" o "domani" (param: bulletin_type)
- Per regioni usa nomi italiani capitalizzati: "Sicilia", "Lombardia", ecc. (param: region_name)
- Per coordinate usa: latitude (float -90 a 90), longitude (float -180 a 180), radius_km (float > 0)

FORMATO OUTPUT OBBLIGATORIO:
TOOL: <nome_tool>
PARAMS: <parametri in formato JSON con nomi esatti>

ESEMPIO 1:
TOOL: get_hotspots_by_intensity
PARAMS: {{"intensity_level": "Alta"}}

ESEMPIO 2:
TOOL: get_flood_zones_by_risk_level
PARAMS: {{"risk_level": "Arancione", "bulletin_type": "oggi"}}

ESEMPIO 3:
TOOL: get_hotspots_by_region
PARAMS: {{"region_name": "Sicilia"}}

DOMANDA UTENTE: {query}

Seleziona il tool più appropriato e estrai i parametri:"""

        # Generate tool selection using Phi-4
        if self.rag_pipeline.llm_model:
            chat = [{"role": "user", "content": prompt}]
            response = self._generate_with_phi4(chat)
            
            # Parse response to extract tool and params
            tool_name, params = self._parse_tool_selection_response(response, query)
        else:
            # Fallback to simple parsing if no LLM
            tool_name, params = self._parse_tool_request(query)
        
        # CRITICAL: Validate and correct parameters to match schema
        params = self._validate_and_correct_params(tool_name, params, query)
        
        return tool_name, params
    
    def _validate_and_correct_params(self, tool_name: str, params: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Validate and correct parameters to match Pydantic schema requirements.
        
        Args:
            tool_name: Name of the tool
            params: Extracted parameters
            query: Original query for re-extraction if needed
            
        Returns:
            Validated and corrected parameters
        """
        logger.info(f"Validating parameters for tool '{tool_name}': {params}")
        query_lower = query.lower()
        
        # Intensity validation (must be Italian)
        if 'intensity' in params:
            valid_intensities = ['Bassa', 'Media', 'Alta', 'Molto Alta', 'Sconosciuta']
            if params['intensity'] not in valid_intensities:
                # Convert English or lowercase to Italian
                intensity_map = {
                    'low': 'Bassa', 'bassa': 'Bassa',
                    'medium': 'Media', 'media': 'Media',
                    'high': 'Alta', 'alta': 'Alta',
                    'very high': 'Molto Alta', 'molto alta': 'Molto Alta',
                    'unknown': 'Sconosciuta', 'sconosciuta': 'Sconosciuta'
                }
                params['intensity'] = intensity_map.get(params['intensity'].lower(), 'Sconosciuta')
        
        # Rename 'intensity' to 'intensity_level' for schema compliance
        if 'intensity' in params:
            params['intensity_level'] = params.pop('intensity')
        
        # Confidence validation (must be Italian)
        if 'confidence' in params:
            valid_confidences = ['Alta', 'Media', 'Bassa', 'Sconosciuta']
            if params['confidence'] not in valid_confidences:
                confidence_map = {
                    'high': 'Alta', 'alta': 'Alta',
                    'nominal': 'Media', 'media': 'Media', 'medium': 'Media',
                    'low': 'Bassa', 'bassa': 'Bassa',
                    'unknown': 'Sconosciuta', 'sconosciuta': 'Sconosciuta'
                }
                params['confidence'] = confidence_map.get(params['confidence'].lower(), 'Sconosciuta')
        
        # Rename 'confidence' to 'confidence_category' for schema compliance
        if 'confidence' in params:
            params['confidence_category'] = params.pop('confidence')
        
        # Sensor validation (must be uppercase)
        if 'sensor' in params:
            params['sensor'] = params['sensor'].upper()
        if 'sensor' in params:
            params['sensor_type'] = params.pop('sensor')
        
        # Satellite validation (must match: Terra, Aqua, N, J1, J2)
        if 'satellite' in params:
            satellite_map = {
                'terra': 'Terra', 'TERRA': 'Terra',
                'aqua': 'Aqua', 'AQUA': 'Aqua',
                'npp': 'N', 'NPP': 'N', 's-npp': 'N', 'suomi': 'N',
                'noaa-20': 'J1', 'noaa 20': 'J1', 'j1': 'J1', 'J1': 'J1',
                'noaa-21': 'J2', 'noaa 21': 'J2', 'j2': 'J2', 'J2': 'J2'
            }
            params['satellite'] = satellite_map.get(params['satellite'], params['satellite'])
        if 'satellite' in params:
            params['satellite_name'] = params.pop('satellite')
        
        # Time period validation (must be Italian: Giorno or Notte)
        if 'time_period' in params:
            time_map = {
                'day': 'Giorno', 'giorno': 'Giorno', 'diurno': 'Giorno',
                'night': 'Notte', 'notte': 'Notte', 'notturno': 'Notte'
            }
            params['time_period'] = time_map.get(params['time_period'].lower(), 'Giorno')
        
        # Risk level validation (must be capitalized Italian: Verde, Giallo, Arancione, Rosso)
        if 'risk_level' in params:
            risk_map = {
                'verde': 'Verde', 'green': 'Verde',
                'giallo': 'Giallo', 'gialla': 'Giallo', 'yellow': 'Giallo',
                'arancione': 'Arancione', 'arancio': 'Arancione', 'orange': 'Arancione',
                'rosso': 'Rosso', 'rossa': 'Rosso', 'red': 'Rosso'
            }
            params['risk_level'] = risk_map.get(params['risk_level'].lower(), 'Verde')
        
        # Bulletin type validation (must be lowercase: oggi or domani)
        if 'bulletin_type' in params:
            if params['bulletin_type'].lower() not in ['oggi', 'domani']:
                params['bulletin_type'] = 'oggi'
            else:
                params['bulletin_type'] = params['bulletin_type'].lower()
        
        # Risk class validation (must be integer 0-3)
        if 'risk_class' in params and not isinstance(params['risk_class'], int):
            try:
                params['risk_class'] = int(params['risk_class'])
            except:
                params['risk_class'] = 0
        
        if 'min_risk_class' in params and not isinstance(params['min_risk_class'], int):
            try:
                params['min_risk_class'] = int(params['min_risk_class'])
            except:
                params['min_risk_class'] = 0
        
        # Pattern parameter (name_pattern -> pattern)
        if 'name_pattern' in params:
            params['pattern'] = params.pop('name_pattern')
        
        # Distance parameter (distance_km -> radius_km for schema compliance)
        if 'distance_km' in params:
            params['radius_km'] = params.pop('distance_km')
        
        logger.info(f"Validated parameters for tool '{tool_name}': {params}")
        return params
    
    def _parse_tool_selection_response(self, response: str, query: str) -> tuple:
        """
        Parse Phi-4 response to extract tool name and parameters.
        
        Args:
            response: Phi-4 generated response
            query: Original user query
            
        Returns:
            Tuple of (tool_name, parameters)
        """
        logger.info(f"Parsing tool selection response: {response[:200]}")
        
        try:
            # Extract tool name
            tool_match = re.search(r'TOOL:\s*(\w+)', response)
            if tool_match:
                tool_name = tool_match.group(1)
            else:
                # Fallback: search for tool names in response or query keywords
                response_lower = response.lower()
                query_lower = query.lower()
                
                # Check for "ultime 24 ore" or "panoramica" - route to statistics
                if 'ultime 24 ore' in query_lower or 'panoramica' in query_lower or 'statistiche' in query_lower:
                    if 'hotspot' in response_lower or 'incend' in query_lower:
                        tool_name = 'get_hotspot_statistics'
                    elif 'flood' in response_lower or 'alluv' in query_lower or 'allert' in query_lower:
                        tool_name = 'get_flood_statistics'
                    else:
                        tool_name = 'get_hotspot_statistics'  # Default to fire stats
                elif 'hotspot' in response_lower or 'incend' in query_lower:
                    tool_name = 'get_hotspot_statistics'
                elif 'flood' in response_lower or 'alluv' in query_lower or 'allert' in query_lower:
                    tool_name = 'get_flood_statistics'
                else:
                    tool_name = 'get_data_summary'
            
            # Extract parameters
            params_match = re.search(r'PARAMS:\s*(\{[^}]*\})', response)
            if params_match:
                params_str = params_match.group(1)
                params = json.loads(params_str)
            else:
                # Extract parameters from query using heuristics
                params = self._extract_params_from_query(query, tool_name)
            
            logger.info(f"Parsed tool: {tool_name}, params: {params}")
            return tool_name, params
            
        except Exception as e:
            logger.warning(f"Error parsing tool selection: {e}, using fallback")
            return self._parse_tool_request(query)
    
    def _extract_params_from_query(self, query: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameters from query based on tool requirements.
        Uses correct Italian values matching Pydantic schemas.
        
        Args:
            query: User query
            tool_name: Selected tool name
            
        Returns:
            Dictionary of parameters with validated values
        """
        params = {}
        query_lower = query.lower()
        
        # Extract region/province/municipality names
        if 'region' in tool_name or 'by_region' in tool_name:
            # Common Italian regions (capitalized properly)
            regions_map = {
                'sicilia': 'Sicilia',
                'calabria': 'Calabria',
                'lombardia': 'Lombardia',
                'lazio': 'Lazio',
                'campania': 'Campania',
                'puglia': 'Puglia',
                'toscana': 'Toscana',
                'piemonte': 'Piemonte',
                'veneto': 'Veneto',
                'emilia-romagna': 'Emilia-Romagna',
                'emilia romagna': 'Emilia-Romagna',
                'sardegna': 'Sardegna',
                'liguria': 'Liguria',
                'marche': 'Marche',
                'abruzzo': 'Abruzzo',
                'friuli': 'Friuli Venezia Giulia',
                'friuli venezia giulia': 'Friuli Venezia Giulia',
                'molise': 'Molise',
                'basilicata': 'Basilicata',
                'umbria': 'Umbria',
                'valle d\'aosta': 'Valle d\'Aosta',
                'trentino': 'Trentino-Alto Adige',
                'trentino-alto adige': 'Trentino-Alto Adige',
                'trentino alto adige': 'Trentino-Alto Adige'
            }
            for region_lower, region_proper in regions_map.items():
                if region_lower in query_lower:
                    params['region_name'] = region_proper
                    break
        
        if 'province' in tool_name:
            # Extract province name (simplified with proper capitalization)
            provinces_map = {
                'milano': 'Milano',
                'roma': 'Roma',
                'napoli': 'Napoli',
                'torino': 'Torino',
                'palermo': 'Palermo',
                'firenze': 'Firenze',
                'bologna': 'Bologna',
                'genova': 'Genova',
                'venezia': 'Venezia',
                'bari': 'Bari',
                'catania': 'Catania',
                'messina': 'Messina',
                'verona': 'Verona',
                'padova': 'Padova',
                'trieste': 'Trieste'
            }
            for prov_lower, prov_proper in provinces_map.items():
                if prov_lower in query_lower:
                    params['province_name'] = prov_proper
                    break
        
        if 'municipality' in tool_name:
            # Extract municipality from query (after "a ", "in ", "di ")
            match = re.search(r'(?:a|in|di)\s+([A-Z][a-zà-ù]+(?:\s+[A-Z][a-zà-ù]+)*)', query)
            if match:
                params['municipality_name'] = match.group(1)
        
        # Extract intensity - MUST use Italian values from schema
        if 'intensity' in tool_name:
            if 'molto alta' in query_lower or 'very high' in query_lower:
                params['intensity_level'] = 'Molto Alta'
            elif 'alta' in query_lower or 'high' in query_lower or 'elevat' in query_lower:
                params['intensity_level'] = 'Alta'
            elif 'media' in query_lower or 'medium' in query_lower or 'moderat' in query_lower:
                params['intensity_level'] = 'Media'
            elif 'bassa' in query_lower or 'low' in query_lower:
                params['intensity_level'] = 'Bassa'
            else:
                params['intensity_level'] = 'Sconosciuta'
        
        # Extract confidence - MUST use Italian values from schema
        if 'confidence' in tool_name:
            if 'alta' in query_lower or 'high' in query_lower:
                params['confidence_category'] = 'Alta'
            elif 'bassa' in query_lower or 'low' in query_lower:
                params['confidence_category'] = 'Bassa'
            elif 'media' in query_lower or 'medium' in query_lower or 'nominal' in query_lower:
                params['confidence_category'] = 'Media'
            else:
                params['confidence_category'] = 'Sconosciuta'
        
        # Extract sensor - MUST be uppercase
        if 'sensor' in tool_name:
            if 'modis' in query_lower:
                params['sensor_type'] = 'MODIS'
            elif 'viirs' in query_lower:
                params['sensor_type'] = 'VIIRS'
        
        # Extract satellite - MUST match schema exactly: Terra, Aqua, N, J1, J2
        if 'satellite' in tool_name:
            if 'terra' in query_lower:
                params['satellite_name'] = 'Terra'
            elif 'aqua' in query_lower:
                params['satellite_name'] = 'Aqua'
            elif 'npp' in query_lower or 's-npp' in query_lower or 'suomi' in query_lower:
                params['satellite_name'] = 'N'
            elif 'noaa-20' in query_lower or 'noaa 20' in query_lower or 'j1' in query_lower:
                params['satellite_name'] = 'J1'
            elif 'noaa-21' in query_lower or 'noaa 21' in query_lower or 'j2' in query_lower:
                params['satellite_name'] = 'J2'
        
        # Extract date
        if 'date' in tool_name or 'by_date' in tool_name:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
            if date_match:
                params['date'] = date_match.group(1)
            elif 'oggi' in query_lower:
                from datetime import datetime
                params['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Extract time of day - MUST use Italian values from schema
        if 'time_of_day' in tool_name:
            if 'giorno' in query_lower or 'diurn' in query_lower or 'day' in query_lower:
                params['time_period'] = 'Giorno'
            elif 'notte' in query_lower or 'nottur' in query_lower or 'night' in query_lower:
                params['time_period'] = 'Notte'
        
        # Extract risk level/class for floods - MUST use Italian capitalized values from schema
        if 'risk_level' in tool_name:
            # Determine bulletin type
            if 'oggi' in query_lower or 'today' in query_lower:
                params['bulletin_type'] = 'oggi'
            elif 'domani' in query_lower or 'tomorrow' in query_lower:
                params['bulletin_type'] = 'domani'
            else:
                params['bulletin_type'] = 'oggi'  # default to today
            
            # Extract risk level - use capitalized Italian values
            risk_level_found = False
            for color, risk_class in RISK_COLOR_MAPPING.items():
                if color in query_lower:
                    # Convert to capitalized Italian color name for risk_level
                    params['risk_level'] = color.capitalize()
                    risk_level_found = True
                    break
            
            if not risk_level_found:
                # Default to Verde if not specified
                params['risk_level'] = 'Verde'
        
        if 'risk_class' in tool_name:
            # First check for color-based risk indicators (arancione, rossa, gialla, verde)
            # Map to numeric risk class 0-3
            risk_class_map = {
                'verde': 0,
                'gialla': 1,
                'giallo': 1,
                'arancione': 2,
                'arancio': 2,
                'rossa': 3,
                'rosso': 3
            }
            
            risk_class_found = False
            for color, risk_num in risk_class_map.items():
                if color in query_lower:
                    params['risk_class'] = risk_num
                    risk_class_found = True
                    break
            
            # If no color found, try direct numeric or risk class names
            if not risk_class_found:
                if 'elevata' in query_lower or 'elevato' in query_lower:
                    params['risk_class'] = 3
                elif 'moderata' in query_lower or 'moderato' in query_lower:
                    params['risk_class'] = 2
                elif 'ordinaria' in query_lower or 'ordinario' in query_lower:
                    params['risk_class'] = 1
                elif 'assente' in query_lower or 'nessun' in query_lower:
                    params['risk_class'] = 0
                else:
                    params['risk_class'] = 0  # default
            
            # Add bulletin_type
            if 'oggi' in query_lower or 'today' in query_lower:
                params['bulletin_type'] = 'oggi'
            elif 'domani' in query_lower or 'tomorrow' in query_lower:
                params['bulletin_type'] = 'domani'
            else:
                params['bulletin_type'] = 'oggi'
            
            # Add default risk_type
            if 'risk_type' not in params:
                params['risk_type'] = 'criticita_class'
            
            # Handle minimum_risk_class variant
            if 'minimum_risk_class' in tool_name and 'risk_class' in params:
                params['min_risk_class'] = params.pop('risk_class')
        
        # Extract zone code
        if 'zone_code' in tool_name:
            code_match = re.search(r'([A-Z][a-z]+-[A-Z0-9]+)', query)
            if code_match:
                params['zone_code'] = code_match.group(1)
            
            # Add bulletin_type
            if 'oggi' in query_lower or 'today' in query_lower:
                params['bulletin_type'] = 'oggi'
            elif 'domani' in query_lower or 'tomorrow' in query_lower:
                params['bulletin_type'] = 'domani'
            else:
                params['bulletin_type'] = 'oggi'
        
        # Extract name pattern
        if 'name_pattern' in tool_name:
            # Try to extract quoted pattern
            pattern_match = re.search(r'["\']([^"\']+)["\']', query)
            if pattern_match:
                params['pattern'] = pattern_match.group(1)
            else:
                # Try to extract after "con" or "con il"
                con_match = re.search(r'con\s+(?:il\s+)?["\']?([A-Za-zàèéìòù]+)["\']?\s+nel\s+nome', query_lower)
                if con_match:
                    params['pattern'] = con_match.group(1).capitalize()
            
            # Add bulletin_type
            if 'oggi' in query_lower or 'today' in query_lower:
                params['bulletin_type'] = 'oggi'
            elif 'domani' in query_lower or 'tomorrow' in query_lower:
                params['bulletin_type'] = 'domani'
            else:
                params['bulletin_type'] = 'oggi'
        
        # Extract coordinates and distance
        if 'within_distance' in tool_name:
            # First try explicit lat/lon
            lat_match = re.search(r'latitudine[:\s]+([0-9.]+)', query_lower)
            lon_match = re.search(r'longitudine[:\s]+([0-9.]+)', query_lower)
            dist_match = re.search(r'(\d+)\s*km', query_lower)
            
            if lat_match:
                params['latitude'] = float(lat_match.group(1))
            if lon_match:
                params['longitude'] = float(lon_match.group(1))
            if dist_match:
                params['radius_km'] = float(dist_match.group(1))
            
            # If no explicit coordinates, try to find city name
            if 'latitude' not in params or 'longitude' not in params:
                # Extract city name (after "da", "a", "di", "vicino a")
                city_match = re.search(r'(?:da|a|di|vicino\s+a|intorno\s+a)\s+([A-Za-zàèéìòù\'\s]+?)(?:\?|$|,|\s+entro|\s+nel)', query, re.IGNORECASE)
                if city_match:
                    city_name = city_match.group(1).strip().lower()
                    if city_name in ITALIAN_CITY_COORDINATES:
                        coords = ITALIAN_CITY_COORDINATES[city_name]
                        params['latitude'] = coords[0]
                        params['longitude'] = coords[1]
                        logger.info(f"Resolved city '{city_name}' to coordinates: {coords}")
            
            # Default radius if not specified
            if 'radius_km' not in params:
                params['radius_km'] = 50.0  # default 50 km
            
            # Add bulletin_type for flood queries
            if 'flood' in tool_name:
                if 'oggi' in query_lower or 'today' in query_lower:
                    params['bulletin_type'] = 'oggi'
                elif 'domani' in query_lower or 'tomorrow' in query_lower:
                    params['bulletin_type'] = 'domani'
                else:
                    params['bulletin_type'] = 'oggi'
        
        # Extract bounding box
        if 'bounding_box' in tool_name:
            lat_matches = re.findall(r'lat[^0-9]*([0-9.]+)', query_lower)
            lon_matches = re.findall(r'lon[^0-9]*([0-9.]+)', query_lower)
            
            if len(lat_matches) >= 2:
                params['min_lat'] = float(min(lat_matches))
                params['max_lat'] = float(max(lat_matches))
            if len(lon_matches) >= 2:
                params['min_lon'] = float(min(lon_matches))
                params['max_lon'] = float(max(lon_matches))
            
            # Add bulletin_type for flood queries
            if 'flood' in tool_name:
                if 'oggi' in query_lower or 'today' in query_lower:
                    params['bulletin_type'] = 'oggi'
                elif 'domani' in query_lower or 'tomorrow' in query_lower:
                    params['bulletin_type'] = 'domani'
                else:
                    params['bulletin_type'] = 'oggi'
        
        # For flood statistics - add bulletin_type
        if tool_name == 'get_flood_statistics':
            if 'oggi' in query_lower or 'today' in query_lower:
                params['bulletin_type'] = 'oggi'
            elif 'domani' in query_lower or 'tomorrow' in query_lower:
                params['bulletin_type'] = 'domani'
            else:
                params['bulletin_type'] = 'oggi'
        
        # For region-based flood queries - add bulletin_type
        if 'flood' in tool_name and 'by_region' in tool_name:
            if 'bulletin_type' not in params:
                if 'oggi' in query_lower or 'today' in query_lower:
                    params['bulletin_type'] = 'oggi'
                elif 'domani' in query_lower or 'tomorrow' in query_lower:
                    params['bulletin_type'] = 'domani'
                else:
                    params['bulletin_type'] = 'oggi'
        
        return params
    
    def _format_tool_result(self, query: str, tool_name: str, tool_params: Dict[str, Any], 
                           tool_result: Any) -> str:
        """
        Format tool result into natural language answer using Phi-4.
        
        Args:
            query: Original user query
            tool_name: Name of executed tool
            tool_params: Parameters passed to tool
            tool_result: Result from tool execution
            
        Returns:
            Natural language answer
        """
        # Build context-aware prompt
        prompt = f"""Sei un assistente specializzato in analisi di dati ambientali su incendi e alluvioni in Italia.

TOOL ESEGUITO: {tool_name}
PARAMETRI: {tool_params}

DATI OTTENUTI:
{tool_result}

DOMANDA ORIGINALE: {query}

ISTRUZIONI:
1. Analizza i dati ottenuti dal tool
2. Rispondi alla domanda originale in modo chiaro e preciso
3. Usa un linguaggio naturale e professionale
4. Includi i dettagli numerici rilevanti dai dati
5. Se ci sono molti risultati, riassumi le informazioni principali

Fornisci la risposta:"""

        chat = [{"role": "user", "content": prompt}]
        answer = self._generate_with_phi4(chat)
        
        return answer
    
    def _format_tool_result_simple(self, tool_result: Any) -> str:
        """
        Simple formatting of tool result without LLM.
        
        Args:
            tool_result: Result from tool execution
            
        Returns:
            Formatted string
        """
        if isinstance(tool_result, dict):
            if 'error' in tool_result:
                return f"Errore: {tool_result['error']}"
            elif 'count' in tool_result:
                return f"Ho trovato {tool_result['count']} risultati. Dettagli: {tool_result}"
            else:
                return f"Risultato: {tool_result}"
        else:
            return f"Risultato: {tool_result}"
    
    def _generate_direct_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer without RAG context
        
        Args:
            query: User query
            
        Returns:
            Result dictionary with answer
        """
        logger.info("Generating direct answer (no RAG)...")
        
        prompt_chat = [
            {
                "role": "system",
                "content": "Sei un esperto in catastrofi naturali, rischi idrogeologici e incendi boschivi in Italia. Fornisci risposte complete e accurate, mantenendole concise."
            },
            {
                "role": "user",
                "content": f"{query}\n\nFornisci una risposta completa ma concisa, includendo tutti i dettagli necessari senza ripetizioni."
            }
        ]
        
        answer = self._generate_with_phi4(prompt_chat)
        
        return {
            'query': query,
            'answer': answer,
            'method': 'direct',
            'num_chunks': 0
        }
    
    def _generate_with_phi4(self, chat: List[Dict[str, str]]) -> str:
        """
        Generate answer using Phi-4 LLM
        
        Args:
            chat: Chat messages in format [{"role": "user", "content": "..."}]
            
        Returns:
            Generated answer
        """
        if not self.rag_pipeline.llm_model or not self.rag_pipeline.llm_tokenizer:
            logger.warning("Phi-4 not available")
            return "LLM non disponibile per generare la risposta."
        
        try:
            # Apply chat template
            chat_formatted = self.rag_pipeline.llm_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            input_tokens = self.rag_pipeline.llm_tokenizer(
                chat_formatted,
                return_tensors="pt"
            ).to(self.rag_pipeline.llm_device)
            
            input_length = input_tokens['input_ids'].shape[1]
            
            # Generate (allow complete answers, temperature=0 for deterministic output)
            output = self.rag_pipeline.llm_model.generate(
                **input_tokens,
                max_new_tokens=512,
                temperature=0,
                do_sample=False,
                pad_token_id=self.rag_pipeline.llm_tokenizer.eos_token_id,
                eos_token_id=self.rag_pipeline.llm_tokenizer.eos_token_id
            )
            
            # Decode only new tokens
            generated_tokens = output[0][input_length:]
            answer = self.rag_pipeline.llm_tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            ).strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating with Phi-4: {e}")
            return f"Errore nella generazione: {str(e)}"
    
    def _parse_tool_request(self, query: str) -> tuple:
        """Parse query to determine tool and parameters"""
        query_lower = query.lower()
        
        # Determine tool
        if "fire" in query_lower or "hotspot" in query_lower or "incend" in query_lower:
            tool_name = "read_fire_data"
            params = {"data_type": "hotspots"}
        elif "flood" in query_lower or "alluv" in query_lower:
            tool_name = "read_flood_data"
            params = {"data_type": "floods"}
        else:
            tool_name = "list_available_data"
            params = {}
        
        return tool_name, params
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def reset(self):
        """Reset conversation history"""
        logger.info("Resetting agent state")
        self.conversation_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics
        
        Returns:
            Dictionary with agent stats
        """
        rag_stats = self.rag_pipeline.get_stats()
        
        return {
            'rag': rag_stats,
            'conversation_length': len(self.conversation_history),
            'model': 'microsoft/Phi-4-mini-instruct'
        }
