# Agent Workflow Documentation

## Streamlined RAG Workflow with Context-Aware Router

### Overview

The agent implements a streamlined workflow that routes queries using a context-aware router that has visibility into all available documents, optionally retrieves context, and uses Phi-4 for final answer generation.

**Key Enhancement**: The router receives summaries of all .md files in `data/processed/` to make precise decisions about when to use RAG.

```
┌─────────────────────────────────────────────────────────────────┐
│              STREAMLINED WORKFLOW (Enhanced Router)              │
└─────────────────────────────────────────────────────────────────┘

    INITIALIZATION
        ↓
┌─────────────────────┐
│  LOAD ALL .md FILES │  → Scans data/processed/**/*.md
│  AS CONTEXT         │     Creates document summary
└─────────────────────┘
        ↓
    USER INPUT
        ↓
┌───────────────────────┐
│   1. CONTEXT-AWARE    │  → Classifies query type
│      ROUTER           │     WITH document context
│  (Qwen3-1.7B)         │     Knows what docs exist
└───────────────────────┘
        ↓
    ┌───────┴────────┬──────────┬─────────┐
    │                │          │         │
RAG_REQUIRED    TOOL_REQUIRED  DIRECT   NONE
(in docs)       (recent data)  (general)(out of scope)
    │                │          │         │
    ↓                ↓          ↓         ↓
┌──────────┐    ┌──────────┐ ┌─────┐  ┌─────────┐
│ 2a. RAG  │    │ 2b. TOOL │ │2c.  │  │2d.      │
│ RETRIEVAL│    │  ACCESS  │ │SKIP │  │GUARDRAIL│
└──────────┘    └──────────┘ └─────┘  └─────────┘
    │                │          │         │
    └────────┬───────┴──────────┘         │
             ↓                             ↓
    ┌────────────────┐              ┌──────────┐
    │ 3. FINAL       │              │ 3. ERROR │
    │    PROMPT      │  → Context   │ MESSAGE  │
    └────────────────┘     + Query  └──────────┘
             ↓
    ┌────────────────┐
    │ 4. PHI-4       │  → microsoft/Phi-4-mini-instruct
    │    GENERATION  │     (14B params, 4-bit quantized)
    └────────────────┘
             ↓
    ┌────────────────┐
    │ 5. FINAL       │
    │    ANSWER      │
    └────────────────┘
             ↓
        USER OUTPUT
```

## Component Details

### 1. Context-Aware Router (Enhanced!)

**Purpose**: Classify incoming queries to determine processing path using document context

**Model**: `Qwen/Qwen3-1.7B` (4-bit quantized, GPU)

**Enhancement**: Router now receives summaries of all .md files in `data/processed/` to make informed decisions.

**Query Types**:
- `RAG`: Answer is contained in available documents (historical data, reports)
- `TOOL`: Needs recent/real-time data not in documents
- `DIRECT`: General domain questions not requiring specific data
- `NONE`: Out of scope / guardrail

**Document Context**:
```python
# Loaded at initialization from data/processed/**/*.md
# Current: 7 documents (~1800 chars)
# - flood/executive-summary_rapporto_dissesto_2018.md
# - flood/executive-summary_rapporto_dissesto_2021.md
# - flood/executive-summary_rapporto_dissesto_2024.md
# - wildfire/report_incendi_2021_ispra.md
# - wildfire/report_incendi_2022_ispra.md
# - wildfire/report_incendi_2023_ispra.md
# - wildfire/report_incendi_2024_ispra.md
```

**Decision Logic**:
```python
# Router sees all available documents in context
routing_prompt = f"""
Hai a disposizione i seguenti documenti:
{document_context}

Query: {query}

Se la risposta è nei documenti → RAG
Se serve dati recenti non nei documenti → TOOL
Se domanda generale senza dati specifici → DIRECT
Altrimenti → NONE
"""

classification = router_model.generate(routing_prompt)
```

**Benefits**:
- ✓ Knows exactly what documents exist
- ✓ More precise RAG decisions
- ✓ Reduced false positives
- ✓ Context-aware classification

### 2a. RAG Retrieval

**Purpose**: Retrieve relevant document chunks for context

**Process**:
1. **Semantic search** retrieves 20 candidate chunks (cast wide net)
2. **Cross-encoder reranker** scores all 20 candidates
3. Select top 5 most relevant chunks after reranking
4. **KeyLLM keyword search** (optional) for additional precision
5. Format chunks into context string
6. Build final prompt with context

**Components**:
- **Embedding Model**: `ibm-granite/granite-embedding-278m-multilingual` (GPU)
- **Reranker**: `jinaai/jina-reranker-v2-base-multilingual` (GPU, cross-encoder)
- **Keyword Model**: `Qwen/Qwen3-0.6B` (GPU, 4-bit for KeyLLM)
- **Vector DB**: ChromaDB
- **Retrieval**: Hybrid (semantic + BM25) with reranking + optional KeyLLM

**KeyLLM Integration**:
- Uses Qwen3-0.6B (0.6B params, ~0.5GB in 4-bit) for intelligent keyword extraction
- Extracts contextually relevant keywords from query + document samples
- Improves keyword search precision with LLM understanding
- Falls back to regex if KeyLLM unavailable

**Why Reranking?**
- Bi-encoder (embedding) is fast but less precise
- Cross-encoder (reranker) is slower but more accurate
- Strategy: Retrieve 20 → Rerank → Select top 5
- Result: +30-40% better precision than direct top-5

### 2b. Tool Access (Forced Tool Calling)

**Purpose**: Access real-time vigilance data with mandatory tool selection

**Available Tools**: 25 vigilance tools across fire hotspots and flood bulletins

**Tool Categories**:
- **Fire Hotspots - Geographic** (3): `get_hotspots_by_region`, `get_hotspots_by_province`, `get_hotspots_by_municipality`
- **Fire Hotspots - Attributes** (4): `get_hotspots_by_intensity`, `get_hotspots_by_confidence`, `get_hotspots_by_sensor`, `get_hotspots_by_satellite`
- **Fire Hotspots - Temporal** (3): `get_hotspots_by_date`, `get_hotspots_by_time_of_day`, `get_hotspots_statistics`
- **Fire Hotspots - Spatial** (2): `get_hotspots_within_distance`, `get_hotspots_in_bounding_box`
- **Flood Bulletins - Geographic** (4): `get_flood_zones_by_region`, `get_flood_zones_by_risk_level`, `get_flood_zones_by_risk_class`, `get_flood_zones_by_minimum_risk`
- **Flood Bulletins - Search** (2): `get_flood_zones_by_zone_code`, `get_flood_zones_by_name_pattern`
- **Flood Bulletins - Spatial** (2): `get_flood_zones_within_distance`, `get_flood_zones_in_bounding_box`
- **Flood Bulletins - Statistics** (1): `get_flood_zones_statistics`
- **Utilities** (4): `get_available_regions`, `get_data_summary`, `list_available_tools`, `get_tool_schema`

**Forced Tool Calling Workflow**:
```
Query → _process_with_tools()
           ↓
    1. Load Tool Descriptions (25 tools)
           ↓
    2. Force Tool Selection with Phi-4
       (Prompt includes all tool descriptions)
           ↓
    3. Parse TOOL: <name> PARAMS: <json>
           ↓
    4. Extract/Validate Parameters
       (Smart extraction from query)
           ↓
    5. Execute Tool via ToolCaller
           ↓
    6. Format Response
       (Natural language + structured data)
```

**Key Principles**:
1. **Mandatory Selection**: Agent MUST select exactly one tool (cannot skip)
2. **Tool-Aware Prompting**: All 25 tool descriptions included in prompt
3. **Intelligent Parameter Extraction**: Extracts region names, dates, coordinates, etc. from natural language
4. **Structured Output**: Returns tool name, input params, and output
5. **Direct Answer**: Tool output used directly (optionally formatted with Phi-4)

**Example Tool Calls**:

*Geographic Query*:
```
Query: "Mostra gli hotspot in Sicilia"
Tool Selected: get_hotspots_by_region
Parameters: {"region_name": "Sicilia"}
Output: {total: 15, hotspots: [...], bounds: [...]}
```

*Temporal Query*:
```
Query: "Incendi rilevati oggi"
Tool Selected: get_hotspots_by_date  
Parameters: {"date": "2025-10-16"}
Output: {total: 8, hotspots: [...]}
```

*Risk Query*:
```
Query: "Zone a rischio alluvioni elevato in Emilia-Romagna"
Tool Selected: get_flood_zones_by_risk_level
Parameters: {"region_name": "Emilia-Romagna", "risk_level": "elevata"}
Output: {total: 3, zones: [...]}
```

**Tool Output Structure**:
```python
{
    'success': bool,              # Execution status
    'total_hotspots': int,        # Count (for fire tools)
    'total_zones': int,           # Count (for flood tools)
    'hotspots': List[dict],       # Fire data records
    'zones': List[dict],          # Flood data records
    'bounds': List[float],        # Geographic bounds [min_lon, min_lat, max_lon, max_lat]
    'statistics': dict            # Aggregated statistics (if requested)
}
```

**Data Sources**:
- Fire Hotspots: `data/vigilance/hotspots/*.parquet` (FIRMS NASA + ISTAT enrichment)
- Flood Bulletins: `data/vigilance/floods/*.parquet` (Italian Civil Protection DPC)

**Note**: Tool calling is optional and requires LangChain. If not available, agent falls back to basic data file access.

### 3. Final Prompt Construction

**With RAG Context**:
```
Basandoti sui seguenti estratti di documenti ISPRA, rispondi alla domanda.

Contesto:
[1] {chunk_1}
[2] {chunk_2}
...

Domanda: {query}

Rispondi in modo conciso e accurato, citando i dati specifici dal contesto.
```

**Without Context** (Direct):
```
{query}
```

### 4. Phi-4 Generation

**Model**: `microsoft/Phi-4-mini-instruct`

**Specifications**:
- **Size**: 14B parameters
- **Quantization**: 4-bit NF4
- **Device**: CUDA (GPU)
- **Memory**: ~2.9 GB VRAM (with 7.5GB limit)
- **Speed**: 15-20 tokens/sec (quantized)

**Note**: All 5 models run on GPU with optimized memory management:
- Router: ~3.2 GB (3.5GB limit)
- Embedding: ~0.3 GB
- Reranker: ~0.5 GB
- KeyLLM (Qwen3-0.6B): ~0.5 GB (4-bit)
- LLM: ~2.9 GB (7.5GB limit)
- **Total**: ~7.4 GB / 8.6 GB (86% utilization)

**Generation Parameters**:
```python
max_new_tokens=150    # Reduced from 512 for faster responses
temperature=0         # Deterministic generation
do_sample=False       # Greedy decoding
```

### 5. Final Answer

**Format**: Clean text response
- No special tokens
- No prompt echo
- Only the generated answer

**Output Structure**:
```python
{
    'query': str,
    'answer': str,
    'query_type': str,  # rag_required, tool_required, direct
    'method': str,      # hybrid, semantic, tool, direct
    'num_chunks': int,  # Number of chunks used (if RAG)
    'timestamp': str
}
```

## Workflow Examples

### Example 1: Historical Query (RAG Required)

**Input**: "Quali sono state le regioni più colpite dagli incendi nel 2021?"

**Processing**:
1. Router → `RAG_REQUIRED` (keywords: "regioni", "incendi", "2021")
2. RAG → Hybrid search retrieves 5 chunks from ISPRA reports
3. Prompt → Context + Query formatted
4. Phi-4 → Generates answer citing specific data
5. Output → "Le regioni più colpite sono: 1. Calabria (13.541 ha)..."

### Example 2: Data File Query (Tool Required)

**Input**: "Mostra i dati recenti sugli hotspot"

**Processing**:
1. Router → `TOOL_REQUIRED` (keywords: "mostra", "dati", "recenti")
2. Tool → `read_fire_data` executes
3. Prompt → Data + Query formatted
4. Phi-4 → Analyzes data and generates answer
5. Output → Analysis of recent hotspot data

### Example 3: General Query (Defaults to RAG)

**Input**: "Che cos'è un incendio boschivo?"

**Processing**:
1. Router → `RAG_REQUIRED` (default, keyword: "incendio")
2. RAG → Retrieves definition/explanation chunks
3. Prompt → Context + Query
4. Phi-4 → Generates comprehensive answer
5. Output → Definition with context from documents

## Key Benefits

1. **Single Model**: All generation uses Phi-4 (no ModelManager needed)
2. **Smart Routing**: Automatic decision on RAG necessity
3. **Hybrid Retrieval**: Best of semantic and keyword search
4. **GPU Optimized**: 4-bit quantization for efficiency
5. **Clean Outputs**: Only answer text, no prompt echo
6. **Italian Support**: Optimized for Italian queries and documents

## Performance

- **Router Classification**: ~0.4-0.8 seconds
- **RAG Retrieval**: ~0.15-0.25 seconds (20 candidates + reranking)
- **RAG Query**: ~5-8 seconds (retrieval + generation)
- **Direct Query**: ~3-5 seconds (generation only)
- **Tool Query**: ~4-6 seconds (tool + generation)

**Generation Speed**: 15-20 tokens/sec (4-bit quantized on GPU)

## Code Structure

```
agent.py
├── Agent class
│   ├── __init__()           # Initialize RAG pipeline with Phi-4
│   ├── route()              # Classify query type
│   ├── process()            # Main workflow orchestration
│   ├── _process_with_tools() # Handle tool queries
│   ├── _generate_direct_answer() # Handle direct queries
│   ├── _generate_with_phi4() # Phi-4 generation wrapper
│   └── get_stats()          # Agent statistics
```

## Configuration

**Environment Variables** (.env):
```bash
LLM_MODEL=microsoft/Phi-4-mini-instruct
LLM_DEVICE=cuda
EMBEDDING_MODEL=ibm-granite/granite-embedding-278m-multilingual
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
DEVICE=cuda

# Memory limits
ROUTER_MAX_MEMORY=3.5GiB
LLM_MAX_MEMORY=7.5GiB

# Generation parameters
MAX_NEW_TOKENS=150
TEMPERATURE=0
DO_SAMPLE=False
```

## Future Enhancements

1. **Conversation Memory**: Multi-turn context awareness
2. **Hybrid Queries**: Combine RAG + Tools in single query
3. **Streaming**: Stream Phi-4 output token-by-token
4. **Citations**: Add source references to answers (partially implemented)
5. **Confidence Scores**: Rate answer confidence based on retrieval scores
6. **Fine-tuning**: Fine-tune models on domain-specific data

---

**Note**: See `docs/CONFIGURATION.md` for detailed system configuration, optimization strategies, and performance metrics.
