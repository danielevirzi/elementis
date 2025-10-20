# Elementis RAG Pipeline Workflow

## 📋 Overview

This document describes the recommended workflow for building the knowledge base for the Elementis RAG pipeline. The workflow is designed for **reproducibility** and **quality control**.

## 🔄 Pipeline Steps

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   PDF       │────▶│   Markdown   │────▶│   Manual    │────▶│   Vector     │
│ Documents   │     │  Extraction  │     │  Cleaning   │     │  Database    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
     PDFs        Docling converter    Remove tables,      LlamaIndex chunking
                                       images, unclear     ChromaDB encoding
```

## 📝 Detailed Workflow

### Step 1: Extract Markdown from PDFs

**Script:** `scripts/extract_markdown.py`

```bash
python scripts/extract_markdown.py
```

**What it does:**
- Scans `data/documents/` for PDF files
- Uses Docling to extract structured content
- Converts to Markdown format
- Saves to `data/processed/{category}/filename.md`

**Output:**
```
data/processed/
├── flood/
│   ├── report_1.md
│   └── report_2.md
└── wildfire/
    ├── report_incendi_2021_ispra.md
    ├── report_incendi_2022_ispra.md
    └── ...
```

### Step 2: Manual Cleaning

**Action:** Edit markdown files manually

**Location:** `data/processed/`

**What to do:**
- Open each `.md` file
- Remove:
  - Tables (converted poorly)
  - Image placeholders
  - Unclear or garbled text
  - Irrelevant sections
  - Formatting artifacts
- Keep:
  - Clean narrative text
  - Key facts and figures
  - Relevant context for Italian wildfires/floods
  - Properly formatted lists

**Example cleaning:**

❌ **Before (raw extraction):**
```markdown
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| Data    | More    | Info    |

![Image: chart_123.png]

Some text... [UNCLEAR SECTION] ...more text
```

✅ **After (cleaned):**
```markdown
Nel 2021, la superficie bruciata ha raggiunto 169.482 ettari, di cui 36.304 ettari erano aree forestali.

Le regioni più colpite sono state:
- Sicilia: 88.714 ettari
- Calabria: 36.179 ettari
- Sardegna: 20.587 ettari
```

### Step 3: Build Vector Database

**Script:** `scripts/build_from_markdown.py`

```bash
python scripts/build_from_markdown.py
```

**What it does:**
- Loads all cleaned markdown files from `data/processed/`
- Chunks text using LlamaIndex SentenceSplitter (512 tokens, 50 overlap)
- Encodes chunks with IBM Granite multilingual embeddings
- Stores in ChromaDB vector database at `data/vector_db/`

**Output:**
```
data/vector_db/
├── chroma.sqlite3
└── [collection folders]
```

## 🎯 Why This Pipeline?

### ✅ Advantages

1. **Reproducibility**: Same markdown → same chunks → same embeddings
2. **Quality Control**: Manual review ensures clean data
3. **Transparency**: You can see and verify what's in the database
4. **Flexibility**: Easy to update specific documents
5. **Better RAG**: Cleaner input = better retrieval = better answers

### ⚠️ Trade-offs

- Requires manual work (Step 2)
- Takes more time than direct PDF processing
- Need to re-clean if extracting again

## 🔧 Usage Examples

### First-time setup

```bash
# Step 1: Extract
python scripts/extract_markdown.py

# Step 2: Clean (manual)
# Edit files in data/processed/

# Step 3: Build
python scripts/build_from_markdown.py
```

### Adding new documents

```bash
# 1. Add PDF to data/documents/
# 2. Extract just that PDF
python scripts/extract_markdown.py

# 3. Clean the new markdown file
# 4. Rebuild (will include new + existing)
python scripts/build_from_markdown.py --force-rebuild
```

### Updating existing documents

```bash
# 1. Edit markdown in data/processed/
# 2. Rebuild
python scripts/build_from_markdown.py --force-rebuild
```

## 📊 Configuration

Key settings in `.env`:

```bash
# Chunking
CHUNK_SIZE=512           # Token size per chunk
CHUNK_OVERLAP=50         # Overlap between chunks

# Embeddings
EMBEDDING_MODEL=ibm-granite/granite-embedding-278m-multilingual
DEVICE=cuda              # or 'cpu'

# Database
CHROMA_DB_PATH=./data/vector_db
CHROMA_COLLECTION_NAME=elementis_documents
```

## 🐛 Troubleshooting

### No markdown files found

```bash
# Make sure PDFs are in data/documents/
ls data/documents/*/*.pdf

# Run extraction
python scripts/extract_markdown.py
```

### Database already exists

```bash
# Force rebuild to use cleaned markdown
python scripts/build_from_markdown.py
# Answer 'y' when prompted, or use --force-rebuild flag
```

### Poor RAG results

1. Review markdown files - are they clean?
2. Check chunk size - too small/large?
3. Verify embeddings are working
4. Test with simple queries first

## 📚 Related Documentation

- [MARKDOWN_WORKFLOW.md](MARKDOWN_WORKFLOW.md) - Original implementation details
- [MARKDOWN_IMPLEMENTATION.md](../MARKDOWN_IMPLEMENTATION.md) - Technical notes
- [README.md](../README.md) - General project overview

## 🎓 Best Practices

1. **Always review extracted markdown** - Docling is good but not perfect
2. **Keep originals** - Don't delete PDFs after extraction
3. **Version control** - Commit cleaned markdown files
4. **Document changes** - Note what you removed/changed
5. **Test incrementally** - Clean and test one document at a time
6. **Backup database** - Save `data/vector_db/` before rebuilding

---

**Last Updated:** October 2025  
**Pipeline Version:** 2.0 (Markdown-first approach)
