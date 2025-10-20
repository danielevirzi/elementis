# Elementis - AI Agent for Natural Catastrophe Analysis

Elementis is an intelligent AI agent designed to analyze natural catastrophe data in Italy, combining RAG (Retrieval-Augmented Generation) capabilities with real-time environmental monitoring tools. Built for local deployment with GPU acceleration.

## ✨ Key Features

- 🧠 **Context-Aware Routing**: Intelligent query classification with document awareness
- 📚 **RAG Pipeline**: ChromaDB vector store with LlamaIndex chunking
- � **25+ Tools**: Pydantic-validated tools for flood/wildfire monitoring
- 🤖 **LangChain Agent**: ReAct agent with structured tool calling
- 🎯 **Local-First**: Runs entirely on local GPU using 4-bit quantized models
- 🌐 **Gradio Interface**: Interactive web UI with real-time workflow visualization
- 🇮🇹 **Italian Language**: Native support for Italian environmental reports

## Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ↓
┌──────────────────────────────────────────┐
│  Context-Aware Router (Qwen3-1.7B)       │
│  • Knows all available documents         │
│  • Routes to RAG/TOOL/DIRECT/NONE        │
└──────┬───────────────────────────────────┘
       │
       ├─[RAG]──→ ChromaDB Retrieval ──┐
       │                                │
       ├─[TOOL]─→ LangChain Agent ─────┤
       │         (25 Pydantic Tools)    │
       │                                ├──→ ┌────────────────┐
       ├─[DIRECT]──→ Skip Context ─────┤    │  Phi-4 (14B)   │
       │                                │    │  Final Answer  │
       └─[NONE]───→ Guardrail ─────────┘    └────────────────┘
```

### Core Components

1. **Context-Aware Router** (`agent.py`)
   - Model: Qwen3-1.7B (4-bit quantized)
   - Loads summaries of all processed documents at startup
   - Routes queries to appropriate pipeline based on content

2. **RAG Pipeline** (`rag_pipeline.py`)
   - Vector DB: ChromaDB with all-MiniLM-L6-v2 embeddings
   - Chunking: LlamaIndex with sentence-aware splitting
   - Documents: 7 ISPRA reports (flood & wildfire, 2018-2024)

3. **Tool System** (`langchain_tools.py`, `tool_caller.py`)
   - 25 tools with Pydantic schemas for type safety
   - LangChain wrappers for agent integration
   - Local GeoJSON/Parquet data sources

4. **LangChain Agent** (`langchain_agent.py`)
   - ReAct pattern for multi-step reasoning
   - HuggingFace inference (Llama-3.2-3B-Instruct)
   - Graceful fallback when API unavailable

5. **Answer Generation**
   - Model: microsoft/Phi-4-mini-instruct (14B, 4-bit)
   - GPU-accelerated with flash attention
   - Streaming support for responsive UI

## Prerequisites

- **Python**: 3.9+ (tested with 3.11)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB recommended (8GB minimum)
- **Storage**: 15GB free space for models and data
- **CUDA**: Compatible CUDA toolkit for PyTorch

## Quick Start

### 1. Environment Setup

**Option A: UV (Recommended - 10x faster!)**
```powershell
# Clone repository
git clone https://github.com/danielevirzi/elementis.git
cd elementis

# Create and activate virtual environment
uv venv
.venv\Scripts\activate  # Windows PowerShell

# Install dependencies
uv pip install -r requirements.txt
```

**Option B: Traditional pip**
```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```powershell
# Copy environment template
copy .env.example .env

# Edit .env with your settings (optional HuggingFace token for LangChain agent)
notepad .env
```

### 3. Build Knowledge Base

**Recommended: Markdown Pipeline (Reproducible)**

```powershell
# Step 1: Extract markdown from PDFs
python scripts/extract_markdown.py

# Step 2: Manual cleanup (IMPORTANT!)
# - Review files in data/processed/flood/ and data/processed/wildfire/
# - Remove tables, images, unclear text
# - Keep only clean narrative text
# - This improves RAG quality significantly

# Step 3: Build vector database from cleaned markdown
python scripts/build_from_markdown.py
```

**Pipeline Flow:**
```
PDFs → Markdown Extraction → Manual Cleaning → LlamaIndex Chunking → ChromaDB Storage
```

See [`docs/PIPELINE_WORKFLOW.md`](docs/PIPELINE_WORKFLOW.md) for details.

### 4. Launch Application

```powershell
python src/app.py
```

Access the interface at: **http://localhost:7860**

## Verification

Run the verification script to check your setup:

```powershell
python scripts/setup.py --verify
```

This validates:
- ✅ Python version and dependencies
- ✅ GPU availability (CUDA)
- ✅ Configuration files
- ✅ Vector database and data files
- ✅ Model accessibility

## Usage

### Web Interface (Gradio)

Start the application:
```powershell
python src/app.py
```

Navigate to `http://localhost:7860` and interact through:

**Main Interface:**
- 💬 Chat input for queries
- 📊 Real-time workflow visualization
- 🔄 Status updates for each processing stage
- 🧠 Toggle "thinking mode" for detailed reasoning traces

**Features:**
- **Auto-initialization**: Agent loads on startup
- **Streaming responses**: Real-time answer generation
- **Workflow tracking**: Visual feedback on routing decisions
- **Error handling**: Graceful degradation with informative messages

### Example Queries

**RAG-Based (Historical Reports):**
```
Quali sono le principali aree a rischio idrogeologico in Italia?
→ Routes to RAG → Retrieves from ISPRA flood reports

Quanti ettari sono stati bruciati in Italia nel 2023?
→ Routes to RAG → Retrieves from ISPRA wildfire reports
```

**Tool-Based (Recent Data):**
```
Mostrami i hotspot di incendio nelle ultime 24 ore
→ Routes to TOOL → LangChain agent calls get_recent_hotspots

Quali sono le allerte idrologiche per oggi?
→ Routes to TOOL → Calls get_flood_bulletins with date filter
```

**Direct Answers (General Knowledge):**
```
Che cos'è il rischio idrogeologico?
→ Routes to DIRECT → Phi-4 generates answer without context

Come si formano gli incendi boschivi?
→ Routes to DIRECT → General knowledge response
```

**Out of Scope:**
```
What's the weather in New York?
→ Routes to NONE → Guardrail response in Italian
```

### Command-Line Tools

**Test RAG pipeline:**
```powershell
python -c "from rag_pipeline import RAGPipeline; rag = RAGPipeline(); print(rag.query('rischio idrogeologico'))"
```

**Test tool calling:**
```powershell
python -c "from tool_caller import ToolCaller; tc = ToolCaller(); print(tc.get_recent_hotspots())"
```

**Test LangChain agent:**
```powershell
python -c "from langchain_agent import VigilanceLangChainAgent; agent = VigilanceLangChainAgent(); print(agent.run('hotspot oggi'))"
```

## Project Structure

```
elementis/
├── src/
│   ├── agent.py              # Main agent with context-aware router
│   ├── app.py                # Gradio web interface
│   ├── rag_pipeline.py       # ChromaDB + LlamaIndex RAG
│   ├── langchain_agent.py    # LangChain ReAct agent
│   ├── langchain_tools.py    # Pydantic tool wrappers
│   ├── tool_caller.py        # Direct tool implementations
│   ├── tool_schema.py        # Pydantic schemas
│   └── models.py             # LLM model loading utilities
│
├── data/
│   ├── documents/            # Original PDFs
│   │   ├── flood/           # ISPRA flood reports
│   │   └── wildfire/        # ISPRA wildfire reports
│   ├── processed/           # Cleaned markdown files
│   │   ├── flood/           # 3 reports (2018, 2021, 2024)
│   │   └── wildfire/        # 4 reports (2021-2024)
│   ├── vector_db/           # ChromaDB storage
│   └── vigilance/           # Real-time monitoring data
│       ├── floods/          # Flood bulletins (GeoJSON)
│       └── hotspots/        # Fire hotspots (GeoJSON)
│
├── scripts/
│   ├── extract_markdown.py  # PDF → Markdown extraction
│   ├── build_from_markdown.py  # Markdown → ChromaDB
│   └── setup.py             # Environment verification
│
├── config/
│   ├── models.yaml          # Model configurations
│   └── rag_settings.yaml    # RAG parameters
│
└── docs/
    ├── AGENT_WORKFLOW.md    # Detailed workflow documentation
    ├── PIPELINE_WORKFLOW.md # Data processing guide
    └── CONFIGURATION.md     # Configuration reference
```

## Data Sources

### Knowledge Base (RAG)

**Flood Reports (ISPRA):**
- `rapporto_dissesto_2018.pdf` → Executive summary (2018 data)
- `rapporto_dissesto_2021.pdf` → Executive summary (2021 data)
- `rapporto_dissesto_2024.pdf` → Executive summary (2024 data)

**Wildfire Reports (ISPRA):**
- `report_incendi_2021_ispra.pdf` → 2021 wildfire statistics
- `report_incendi_2022_ispra.pdf` → 2022 wildfire statistics
- `report_incendi_2023_ispra.pdf` → 2023 wildfire statistics
- `report_incendi_2024_ispra.pdf` → 2024 wildfire statistics

**Total**: 7 documents, ~50 pages processed, 200+ semantic chunks

### Real-Time Data (Tools)

**Flood Bulletins:**
- Source: `data/vigilance/floods/*.geojson`
- Updates: Daily bulletins (oggi/domani)
- Coverage: Italian regions and provinces
- Metadata: Alert levels, affected areas

**Fire Hotspots:**
- Source: `data/vigilance/hotspots/*.geojson`
- Updates: Last 24 hours
- Coverage: Italy and Mediterranean
- Metadata: Coordinates, intensity, satellite source

## Configuration

### Model Settings (`config/models.yaml`)

```yaml
router_model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"  # Context-aware routing
  load_in_4bit: true
  device: "cuda"
  max_new_tokens: 50

generator_model:
  name: "microsoft/Phi-4"              # Final answer generation
  load_in_4bit: true
  device: "cuda"
  max_new_tokens: 2048
  temperature: 0.7

langchain_model:
  name: "meta-llama/Llama-3.2-3B-Instruct"  # Tool calling agent
  temperature: 0.1
  max_tokens: 2048
```

### RAG Settings (`config/rag_settings.yaml`)

```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"

chunking:
  chunk_size: 512           # LlamaIndex sentence-aware chunks
  chunk_overlap: 50

retrieval:
  top_k: 5                  # Number of chunks to retrieve
  similarity_threshold: 0.3  # Minimum cosine similarity

vector_db:
  persist_directory: "data/vector_db"
  collection_name: "elementis_docs"
```

### Environment Variables (`.env`)

```bash
# Optional: HuggingFace token for LangChain agent
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# Model override (optional)
HUGGINGFACE_MODEL=meta-llama/Llama-3.2-3B-Instruct

# Logging
LOG_LEVEL=INFO
```

**Note**: HuggingFace token is optional. The agent works without it using fallback mechanisms.

## Development

### Testing

**Unit Tests:**
```powershell
pytest tests/
```

**Component Testing:**
```powershell
# Test router
python -c "from agent import Agent; a = Agent(); print(a.route_query('rischio alluvioni'))"

# Test RAG retrieval
python -c "from rag_pipeline import RAGPipeline; r = RAGPipeline(); print(r.retrieve('incendi 2023'))"

# Test tool system
python -c "from tool_caller import ToolCaller; t = ToolCaller(); print(t.list_tools())"

# Test LangChain agent
python -c "from langchain_agent import VigilanceLangChainAgent; a = VigilanceLangChainAgent(); print(a.run('allerte oggi'))"
```

**Interactive Testing:**
```powershell
# Launch Jupyter for interactive exploration
jupyter notebook

# Open notebooks/exploration.ipynb
```

### Adding New Documents

1. **Add PDF to `data/documents/flood/` or `data/documents/wildfire/`**

2. **Extract markdown:**
   ```powershell
   python scripts/extract_markdown.py
   ```

3. **Clean markdown in `data/processed/`**
   - Remove tables and images
   - Fix formatting issues
   - Keep only relevant narrative text

4. **Rebuild vector database:**
   ```powershell
   python scripts/build_from_markdown.py
   ```

5. **Verify in UI:**
   - Restart app
   - Query for new content
   - Check workflow shows RAG retrieval

### Adding New Tools

1. **Define Pydantic schema in `src/tool_schema.py`:**
   ```python
   class MyNewToolInput(BaseModel):
       param1: str = Field(description="Parameter description")
   ```

2. **Implement tool in `src/tool_caller.py`:**
   ```python
   def my_new_tool(self, param1: str) -> Dict[str, Any]:
       """Tool implementation"""
       return {"result": "data"}
   ```

3. **Create LangChain wrapper in `src/langchain_tools.py`:**
   ```python
   @tool(args_schema=MyNewToolInput)
   def my_new_tool_wrapper(param1: str) -> str:
       """Tool for LangChain agent"""
       caller = ToolCaller()
       return json.dumps(caller.my_new_tool(param1))
   ```

4. **Add to tool list in `langchain_tools.py`:**
   ```python
   def get_all_vigilance_tools():
       return [..., my_new_tool_wrapper]
   ```

### Logging and Debugging

**Enable debug logging:**
```python
# In src/utils.py or .env
LOG_LEVEL=DEBUG
```

**View logs:**
```powershell
# Real-time logs
Get-Content logs/app.log -Wait -Tail 50

# Filter specific component
Select-String -Path logs/app.log -Pattern "router"
```

**Workflow debugging:**
- Enable "Mostra Ragionamento" in Gradio UI
- Check "Stato Workflow" panel for routing decisions
- Review "Dettagli Flusso" for step-by-step execution

## Technologies & Models

### Core Technologies

- **PyTorch** 2.5+ with CUDA support for GPU acceleration
- **Transformers** 4.47+ for model loading and inference
- **LangChain** 0.3+ for agent orchestration
- **ChromaDB** 0.5+ for vector storage
- **LlamaIndex** for semantic chunking
- **Gradio** 5.0+ for web interface
- **Pydantic** 2.0+ for data validation

### Models

**Router (Classification):**
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Size: 1.7B parameters (4-bit quantized → ~1GB VRAM)
- Purpose: Query classification with document context
- Speed: ~50ms per routing decision

**Generator (Final Answers):**
- Model: `microsoft/Phi-4`
- Size: 14B parameters (4-bit quantized → ~8GB VRAM)
- Purpose: Final answer generation
- Features: Flash attention, Italian language support

**Embeddings:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Size: 22M parameters
- Dimension: 384
- Purpose: Document and query embeddings

**LangChain Agent (Optional):**
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Deployment: HuggingFace Inference API
- Purpose: Tool calling for real-time data
- Fallback: Works without API key via mock responses

### Performance

**Hardware Requirements:**
- GPU: 8GB VRAM minimum (RTX 3060)
- RAM: 16GB system memory
- Storage: 15GB for models and data

**Typical Response Times:**
- Routing: 50-100ms
- RAG retrieval: 200-300ms
- Answer generation: 2-5 seconds (streaming)
- Tool execution: 100-500ms

**Throughput:**
- Concurrent users: 1-3 (single GPU)
- Queries per minute: 10-15

## Documentation

- [`docs/AGENT_WORKFLOW.md`](docs/AGENT_WORKFLOW.md) - Detailed workflow and routing logic
- [`docs/PIPELINE_WORKFLOW.md`](docs/PIPELINE_WORKFLOW.md) - Data processing pipeline
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) - Configuration reference
- [`docs/TOOLS.md`](docs/TOOLS.md) - Complete tool documentation

## Troubleshooting

**GPU Not Detected:**
```powershell
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Out of Memory:**
- Reduce `max_new_tokens` in `config/models.yaml`
- Close other GPU applications
- Use smaller batch sizes
- Consider using CPU fallback (slower)

**Model Download Issues:**
```powershell
# Pre-download models
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-4')"
```

**ChromaDB Errors:**
```powershell
# Rebuild vector database
Remove-Item data/vector_db -Recurse -Force
python scripts/build_from_markdown.py
```

**LangChain Agent Not Working:**
- Check HuggingFace token in `.env`
- Agent gracefully falls back without token
- Verify with: `python -c "import os; print(os.getenv('HUGGINGFACE_TOKEN'))"`

## Roadmap

**Completed ✅**
- [x] Context-aware router with document summaries
- [x] RAG pipeline with LlamaIndex + ChromaDB
- [x] 25 Pydantic-validated tools
- [x] LangChain ReAct agent integration
- [x] Gradio UI with workflow visualization
- [x] Local GPU deployment (4-bit quantization)
- [x] Italian language support

**In Progress 🚧**
- [ ] Multi-turn conversation with memory
- [ ] Advanced prompt engineering for better routing
- [ ] Performance benchmarking suite
- [ ] Docker containerization

**Future Enhancements 🔮**
- [ ] Real-time data updates (automated scraping)
- [ ] Fine-tuned router model on domain queries
- [ ] API endpoints for integration
- [ ] Multi-language support (English, French)
- [ ] Add more documents to vector DB

