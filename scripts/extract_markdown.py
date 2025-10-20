"""
Script to extract markdown from PDF documents using Docling

Pipeline Step 1: PDF → Markdown Extraction
This script extracts markdown from PDFs and saves them to data/processed/
Next steps:
- Manually clean the markdown files
- Run build_from_markdown.py to build the knowledge base
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline import RAGPipeline
from utils import setup_logging
from dotenv import load_dotenv


def main():
    """Extract markdown from PDF documents (Step 1 of pipeline)"""
    print("=" * 60)
    print("Elementis - Extract Markdown from PDFs (Step 1)")
    print("=" * 60)
    print("\nPipeline:")
    print("  1. Extract markdown from PDFs (this script)")
    print("  2. Manually clean markdown files")
    print("  3. Build knowledge base (build_from_markdown.py)")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Setup logging
    logger = setup_logging()
    
    print("\nInitializing RAG pipeline...")
    try:
        rag = RAGPipeline()
    except Exception as e:
        print(f"[ERROR] Error initializing RAG pipeline: {e}")
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return 1
    
    # Check for documents
    docs_path = Path("data/documents")
    pdf_files = list(docs_path.rglob("*.pdf"))
    
    print(f"\nFound {len(pdf_files)} PDF files in {docs_path}")
    
    if not pdf_files:
        print("[WARNING]  No PDF files found!")
        print(f"   Please add PDF documents to {docs_path}")
        return 1
    
    # List files
    for i, pdf_file in enumerate(pdf_files, 1):
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        relative_path = pdf_file.relative_to(docs_path)
        print(f"   {i}. {relative_path} ({size_mb:.2f} MB)")
    
    print("\n[INFO] Extracting markdown from PDFs...")
    print("This will save markdown files to data/processed/\n")
    
    # Process each document and save markdown
    processed_path = Path("data/processed")
    success_count = 0
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file.name}...")
            doc = rag.process_document(pdf_file, save_markdown=True)
            success_count += 1
            
            # Show saved location
            category_folder = processed_path / pdf_file.parent.name
            markdown_filename = pdf_file.stem + ".md"
            markdown_path = category_folder / markdown_filename
            print(f"   [OK] Saved to {markdown_path}")
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            logger.error(f"Failed to process {pdf_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Extraction complete! {success_count}/{len(pdf_files)} files processed")
    print("=" * 60)
    
    print(f"\n[INFO] Markdown files saved to: {processed_path.absolute()}")
    print("\n[INFO] Next steps:")
    print("   1. Review and clean the markdown files in data/processed/")
    print("      - Remove tables, images, and unclear text")
    print("      - Keep only clean text for RAG pipeline")
    print("   2. Run: python scripts/build_from_markdown.py")
    print("      - This will chunk and encode to vector database")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
