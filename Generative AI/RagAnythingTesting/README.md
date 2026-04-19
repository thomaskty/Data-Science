# RAG-Anything Testing
Minimal implementation to test RAG-Anything multimodal document processing capabilities.
A simplified wrapper (`SimpleRAG`) that:
- Processes multimodal documents (PDF, images, tables, equations)
- Builds knowledge graphs from processed content
- Enables querying with GPT-4o-mini models
- Caches processed data to avoid reprocessing

## Configuration (.env)
```env
API_KEY=your-openai-key
WORKING_DIR=rag_test_storage
LLM_MODEL=gpt-4o-mini
VISION_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
PARSER=mineru
PARSE_METHOD=auto
ENABLE_IMAGE=True
ENABLE_TABLE=True
ENABLE_EQUATION=True
CHUNK_SIZE=1200
CHUNK_OVERLAP=100
MAX_TOKENS=4000
```

## Usage
```python
from core import SimpleRAG

# Initialize
rag = SimpleRAG()
await rag.initialize()

# Process document (first time only)
await rag.process("document.pdf")

# Query (works with cached data)
answer = await rag.query("your question here")
```

## Key Features Tested
1. **Multimodal Processing**: Text + Equations + Tables + Images
2. **Knowledge Graph**: Dual-graph construction (cross-modal + text-based)
3. **Smart Caching**: Skip reprocessing on subsequent runs
4. **Hybrid Retrieval**: Semantic + structural navigation

## Models Used
- **LLM**: gpt-4o-mini (200K TPM, cost-effective)
- **Vision**: gpt-4o-mini (for image processing)
- **Embeddings**: text-embedding-3-large/small
- **Processing**: ~30-60 seconds (one-time)
- **Queries**: ~2-5 seconds
- **Accuracy**: Successfully extracted complex table with equations
