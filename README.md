# Elementis - AI Agent for Natural Catastrophe Analysis

Elementis is an intelligent AI agent designed to analyze natural catastrophe data, combining RAG (Retrieval-Augmented Generation) capabilities with real-time data tools for environmental monitoring and risk assessment.

## ✨ New: LangChain + Pydantic Integration

**Now with intelligent tool selection powered by LangChain!**
- 🔧 **25 Validated Tools**: Pydantic schemas for type-safe inputs
- 🤖 **ReAct Agent**: Automatic tool selection and multi-step reasoning
- 🔄 **Graceful Fallback**: Works with or without API keys
- 📖 **Self-Documenting**: Tools automatically generate their own docs

See [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for full integration details!

## Features

- 🤖 **Agentic AI**: Router and orchestrator pattern for intelligent task handling
- 📚 **RAG Pipeline**: Document processing with Docling and ChromaDB for knowledge retrieval
- 🔧 **Tool Integration**: 25+ tools with Pydantic validation and LangChain wrappers
- 🎯 **Local-First**: Runs entirely on local GPU using quantized models
- 🌐 **Gradio Interface**: User-friendly web interface for interactions
- 🇮🇹 **Italian Language**: Full support for Italian environmental monitoring

## Architecture

```
User Query → Router (Qwen3-1.7B) → [RAG | TOOL | DIRECT] → Response
                                        ↓
                                   LangChain Agent (optional)
                                        ↓
                                   25 Pydantic Tools
```

### Components

- **Agent Router**: Determines whether to use RAG or tools
- **RAG Pipeline**: Processes documents using Docling and retrieves relevant context
- **Tool Caller**: Accesses local GeoJSON/Parquet data files
- **Models**: Local LLM integration via Hugging face

## Prerequisites

- Python 3.9+
- Ollama installed and running locally
- At least 8GB RAM (16GB recommended)
- 10GB free disk space

## Installation

### Quick Setup with UV (Recommended - 10x faster!)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd elementis
   ```

2. **Create virtual environment with UV**
   ```bash
   uv venv
   ```

3. **Activate virtual environment**
   ```bash
   .venv\Scripts\activate  # Windows PowerShell
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

4. **Install dependencies with UV**
   ```bash
   uv pip install -r requirements.txt
   ```

### Alternative: Traditional Setup with pip

1. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Mac/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   ```bash
   copy .env.example .env  # Windows
   # cp .env.example .env  # Mac/Linux
   # Edit .env with your configurations
   ```

6. **Start Ollama** (in a separate terminal)
   ```bash
   ollama serve
   ```

7. **Download and set up Ollama models**
   ```bash
   python scripts/download_models.py
   ```

8. **Build knowledge base using the recommended pipeline**

   **Recommended Pipeline (For Reproducibility):**
   
   This workflow ensures clean, reproducible document processing:
   
   ```bash
   # Step 1: Extract markdown from PDFs
   python scripts/extract_markdown.py
   
   # Step 2: Manually clean markdown files in data/processed/
   # - Remove tables, images, and unclear text
   # - Keep only clean text for RAG pipeline
   # - Edit the .md files to improve quality
   
   # Step 3: Build vector database from cleaned markdown
   python scripts/build_from_markdown.py
   ```
   
   **Pipeline Flow:**
   ```
   PDFs → Extract Markdown → Manual Cleaning → Chunk with LlamaIndex → Encode to ChromaDB
   ```
   
   See [Markdown Workflow Guide](docs/MARKDOWN_WORKFLOW.md) for details.

## Verify Setup

Run the verification script:
```bash
python quick_start.py
```

This checks:
- ✅ Python version
- ✅ Virtual environment
- ✅ Dependencies
- ✅ Configuration files
- ✅ Directory structure
- ✅ Ollama connection

## Usage

### Running the Application

```bash
python src/app.py
```

The Gradio interface will be available at `http://localhost:7860`

### Example Queries

- "What are the main flood risk areas in Italy according to ISPRA?"
- "Show me recent fire hotspots in the Mediterranean region"
- "Analyze flood data from the last month"
- "What does the documentation say about seismic risk assessment?"

## Data Structure

Place your data files in the appropriate directories:

- `data/documents/`: PDF documents (ISPRA reports, etc.)
- `data/vigilance/hotspots/`: Fire hotspot data (GeoJSON/Parquet)
- `data/vigilance/floods/`: Flood monitoring data (GeoJSON/Parquet)

## Configuration

Edit configuration files in `config/`:

- `models.yaml`: LLM model settings
- `rag_settings.yaml`: RAG pipeline parameters

## Development

### Running Tests

```bash
# Test RAG pipeline
jupyter notebook notebooks/test_rag.ipynb

# Test tool calling
jupyter notebook notebooks/test_tools.ipynb
```

### Project Structure

See the full project structure in `docs/setup_guide.md`

## Technologies Used

- **Ollama**: Local LLM inference
- **Docling**: Advanced document processing
- **ChromaDB**: Vector database for embeddings
- **Gradio**: Web interface
- **LangChain**: Agent framework
- **GeoPandas**: Geospatial data processing

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

[Your License Here]

## Contact

[Your Contact Information]
