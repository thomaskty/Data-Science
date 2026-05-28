# Multi-Modal Hybrid RAG Learning Project

A practical multi-modal RAG system for learning advanced retrieval techniques with PDF documents.

---

### Database Setup

Create a database named `hybridRag` with pgvector extension in postgres database

---

```bash
# First time (creates tables)
python main_ingest.py --init-db --pdf sample_data/your_file.pdf

# Subsequent documents
python main_ingest.py --pdf sample_data/another_file.pdf
```

**Chunking options:**
- `--strategy recursive` (default, recommended)
- `--strategy semantic` (paragraph-based)
- `--strategy fixed` (fixed size with overlap)

### Query the Document
```bash
# Single question
python main_query.py --query "What is the total salary?"

# Interactive mode (recommended)
python main_query.py --interactive
```

## 📂 Project Structure
```commandline
├── config
│   └── settings.py
├── .env
├── .gitignore
├── ingestion
│   ├── chunker.py
│   ├── embedder.py
│   └── parser.py
├── main_ingest.py
├── main_query.py
├── readme.md
├── requirements.txt
├── storage
│   └── schema.sql
└── stores
    ├── graph_store.py
    ├── media_store.py
    └── postgres_store.py
```

---

## 🔍 What Happens During Ingestion
```
PDF → Parse (text + images) → Chunk text → Generate embeddings → 
Store in Postgres + Build knowledge graph → Done!
```

**Stores created:**
- **Text chunks** with vector embeddings
- **Images** with GPT-4V descriptions
- **Knowledge graph** (document → chunks → media)
- **Relationships** between all entities

---

**In pgAdmin, run:**
```sql
-- See all documents
SELECT * FROM documents;

-- See text chunks
SELECT chunk_id, page_number, LEFT(text_content, 100) as preview 
FROM chunks;

-- See images with descriptions
SELECT media_id, page_number, file_path, LEFT(caption, 100) as description 
FROM media;

-- See relationships
SELECT * FROM relationships;
```

## ⚙️ Configuration

Edit `config/settings.py` for:

**Models (cost-optimized):**
```python
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
PROCESS_IMAGES = True  # Set False to skip images
```
