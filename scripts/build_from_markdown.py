"""
Script to build knowledge base from cleaned markdown files

Pipeline Step 3: Chunk Markdown → Encode to Vector Database
Prerequisites:
- Step 1: Run extract_markdown.py to extract markdown from PDFs
- Step 2: Manually clean markdown files in data/processed/
- Step 3: Run this script to chunk and encode to ChromaDB
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline import RAGPipeline
from utils import setup_logging
from dotenv import load_dotenv


def main():
    """Build knowledge base from cleaned markdown (Step 3 of pipeline)"""
    print("=" * 60)
    print("Elementis - Build Knowledge Base (Step 3)")
    print("=" * 60)
    print("\nPipeline:")
    print("  1. Extract markdown from PDFs [DONE]")
    print("  2. Manually clean markdown files [DONE]")
    print("  3. Build knowledge base (this script)")
    print("     - Chunk markdown with LlamaIndex")
    print("     - Encode chunks with embeddings")
    print("     - Store in ChromaDB vector database")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Setup logging
    logger = setup_logging()
    
    print("\nInitializing RAG pipeline...")
    try:
        rag = RAGPipeline()
    except Exception as e:
        print(f"[Error] Error initializing RAG pipeline: {e}")
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return 1
    
    # Check for markdown files
    processed_path = Path("data/processed")
    markdown_files = list(processed_path.rglob("*.md"))
    
    print(f"\n[INFO] Found {len(markdown_files)} markdown files in {processed_path}")
    
    if not markdown_files:
        print("[Warning]  No markdown files found!")
        print(f"   Please run: python scripts/extract_markdown.py first")
        print(f"   Then manually clean the markdown files")
        return 1
    
    # List files
    for i, md_file in enumerate(markdown_files, 1):
        size_kb = md_file.stat().st_size / 1024
        relative_path = md_file.relative_to(processed_path)
        print(f"   {i}. {relative_path} ({size_kb:.1f} KB)")
    
    print("\n[INFO] Building knowledge base from cleaned markdown...")
    print("This may take several minutes...\n")
    
    try:
        rag.build_knowledge_base_from_markdown(force_rebuild=True)
        
        # Show statistics
        stats = rag.get_stats()
        print("\n[INFO] Knowledge base built successfully!")
        print(f"\nStatistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Database path: {stats['db_path']}")
        
    except Exception as e:
        print(f"\n[Error] Error building knowledge base: {e}")
        logger.error(f"Failed to build knowledge base: {e}")
        return 1
    
    # Test query
    print("\n[INFO] Testing with sample query...")
    test_query = "Quali sono le principali catastrofi naturali?"
    
    try:
        results = rag.query(test_query, top_k=3)
        print(f"   Found {len(results)} results")
        
        if results:
            print(f"\n   Top result (score: {results[0]['score']:.4f}):")
            print(f"   Category: {results[0]['metadata'].get('category', 'N/A')}")
            print(f"   File: {results[0]['metadata'].get('filename', 'N/A')}")
            print(f"   Preview: {results[0]['content'][:200]}...")
    except Exception as e:
        print(f"   [Error]  Test query failed: {e}")
    
    print("\n" + "=" * 60)
    print("Knowledge base is ready!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
