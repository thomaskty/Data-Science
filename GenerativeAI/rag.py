import os, re, math, time, json, requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment first.")

BASE_URL = "https://api.openai.com/v1"
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-5"

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

DOCUMENTS = [
    {
        "doc_id": "doc_1",
        "title": "RAG Basics",
        "version": "v1",
        "date": "2024-01-10",
        "type": "note",
        "text": (
            "Retrieval-Augmented Generation (RAG) uses external documents to ground LLM responses. "
            "A typical pipeline: chunk documents, embed chunks, store vectors, retrieve top-k by similarity, "
            "optionally re-rank, assemble context, then generate."
        ),
    },
    {
        "doc_id": "doc_2",
        "title": "Chunking Strategies",
        "version": "v2",
        "date": "2024-02-20",
        "type": "note",
        "text": (
            "Chunking strategies include fixed-size, sentence-based, paragraph-based, overlap, and semantic chunking. "
            "Overlaps help preserve context across boundaries."
        ),
    },
    {
        "doc_id": "doc_3",
        "title": "Hybrid Retrieval",
        "version": "v1",
        "date": "2024-03-02",
        "type": "note",
        "text": (
            "Hybrid retrieval combines dense vector search with keyword search like BM25. "
            "Results can be merged using Reciprocal Rank Fusion (RRF)."
        ),
    },
]

# ----------------------------
# Helpers
# ----------------------------
def openai_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload))
    if not r.ok:
        raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text}")
    return r.json()

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Embeddings endpoint: /v1/embeddings ([platform.openai.com](https://platform.openai.com/docs/api-reference/embeddings/create%20.pdf?utm_source=openai))
    payload = {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"}
    data = openai_post("/embeddings", payload)
    # data["data"] is a list of embedding objects with "embedding" vectors
    return [item["embedding"] for item in data["data"]]

def responses_create(input_text: str, instructions: str = None) -> str:
    # Responses endpoint: /v1/responses ([platform.openai.com](https://platform.openai.com/docs/api-reference/responses/compact/?utm_source=openai))
    payload = {"model": GEN_MODEL, "input": input_text}
    if instructions:
        payload["instructions"] = instructions
    resp = openai_post("/responses", payload)

    # Extract output text robustly
    # The API returns output items; text content has type "output_text". ([platform.openai.com](https://platform.openai.com/docs/api-reference/responses/compact/))
    output_text = []
    for item in resp.get("output", []):
        content = item.get("content", [])
        for c in content:
            if c.get("type") == "output_text":
                output_text.append(c.get("text", ""))
    if output_text:
        return "\n".join(output_text).strip()

    # Fallback (some clients expose output_text directly)
    return resp.get("output_text", "").strip()

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_words=120, overlap=30) -> List[str]:
    # Simple sentence-ish split, then pack into chunks
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], []
    for s in sentences:
        words = s.split()
        if len(" ".join(current + words).split()) <= max_words:
            current += words
        else:
            if current:
                chunks.append(" ".join(current))
            # start new chunk with overlap
            overlap_words = current[-overlap:] if overlap > 0 else []
            current = overlap_words + words
    if current:
        chunks.append(" ".join(current))
    return chunks

def build_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for d in docs:
        parts = chunk_text(d["text"])
        for i, part in enumerate(parts):
            chunks.append({
                "chunk_id": f'{d["doc_id"]}_c{i}',
                "doc_id": d["doc_id"],
                "text": part,
                "meta": {k: d[k] for k in d if k != "text"},
            })
    return chunks

# ----------------------------
# BM25 (minimal)
# ----------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25Index:
    def __init__(self, docs: List[str]):
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.df = {}
        for doc in self.docs:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - df + 0.5)/(df + 0.5) + 1) for t, df in self.df.items()}

    def score(self, query: str, k1=1.5, b=0.75) -> List[float]:
        q = tokenize(query)
        scores = []
        for doc in self.docs:
            score = 0.0
            dl = len(doc)
            tf = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            for t in q:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = tf[t] + k1 * (1 - b + b * dl / self.avgdl)
                score += idf * (tf[t] * (k1 + 1)) / denom
            scores.append(score)
        return scores

# ----------------------------
# Retrieval
# ----------------------------
def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def dense_retrieve(query: str, chunk_vectors: List[List[float]], topk=5):
    qvec = embed_texts([query])[0]
    # OpenAI embeddings are normalized; dot product ~= cosine similarity. ([help.openai.com](https://help.openai.com/id-id/articles/6824809-embeddings-faq?utm_source=openai))
    scores = [dot(qvec, v) for v in chunk_vectors]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:topk], scores

def hybrid_retrieve(query: str, chunk_texts: List[str], chunk_vectors: List[List[float]], topk=5, rrf_k=60):
    dense_rank, dense_scores = dense_retrieve(query, chunk_vectors, topk=len(chunk_texts))
    bm25 = BM25Index(chunk_texts)
    bm25_scores = bm25.score(query)
    bm25_rank = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

    # Reciprocal Rank Fusion
    rrf = [0.0] * len(chunk_texts)
    for rank_list in (dense_rank, bm25_rank):
        for r, idx in enumerate(rank_list):
            rrf[idx] += 1.0 / (rrf_k + r + 1)

    ranked = sorted(range(len(rrf)), key=lambda i: rrf[i], reverse=True)
    return ranked[:topk], rrf, dense_scores, bm25_scores

# ----------------------------
# Reranking (LLM-based scoring)
# ----------------------------
def rerank_llm(query: str, chunks: List[Dict[str, Any]], topn=5):
    scored = []
    for c in chunks:
        prompt = (
            "Score the relevance of the chunk to the query from 0 to 100.\n"
            "Return only a number.\n\n"
            f"Query: {query}\n"
            f"Chunk: {c['text']}"
        )
        score_text = responses_create(prompt)
        try:
            score = float(re.findall(r"\d+(\.\d+)?", score_text)[0])
        except Exception:
            score = 0.0
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:topn]]

# ----------------------------
# Context assembly + prompting
# ----------------------------
def assemble_context(chunks: List[Dict[str, Any]], max_chars=1500):
    seen = set()
    out = []
    total = 0
    for c in chunks:
        if c["chunk_id"] in seen:
            continue
        snippet = f"[{c['chunk_id']}] {c['text']}"
        if total + len(snippet) > max_chars:
            break
        out.append(snippet)
        total += len(snippet)
        seen.add(c["chunk_id"])
    return "\n".join(out)

SYSTEM_PROMPT = (
    "You are a grounded assistant. Use only the provided context to answer. "
    "If the answer is not in context, say you don't know."
)

INSTRUCTIONS = (
    "Answer concisely. If you use facts from context, cite the chunk ids in brackets."
)

def answer_query(query: str, context: str) -> str:
    user_input = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )
    return responses_create(user_input, instructions=f"{SYSTEM_PROMPT}\n{INSTRUCTIONS}")

# ----------------------------
# Monitoring (tiny)
# ----------------------------
EVENTS = []

def log_event(name: str, data: Dict[str, Any]):
    EVENTS.append({"event": name, "ts": time.time(), **data})

# ----------------------------
# Build pipeline
# ----------------------------
chunks = build_chunks(DOCUMENTS)
chunk_texts = [c["text"] for c in chunks]

t0 = time.time()
chunk_vectors = embed_texts(chunk_texts)
log_event("embed_chunks", {"n_chunks": len(chunks), "latency_s": time.time() - t0})

# ----------------------------
# Demo query
# ----------------------------
def run_demo(query: str):
    t0 = time.time()
    ranked_ids, rrf_scores, dense_scores, bm25_scores = hybrid_retrieve(
        query, chunk_texts, chunk_vectors, topk=6
    )
    retrieved = [chunks[i] for i in ranked_ids]
    log_event("retrieve", {"query": query, "topk": [c["chunk_id"] for c in retrieved], "latency_s": time.time() - t0})

    # Rerank top 6 -> top 3
    t1 = time.time()
    reranked = rerank_llm(query, retrieved, topn=3)
    log_event("rerank", {"query": query, "topk": [c["chunk_id"] for c in reranked], "latency_s": time.time() - t1})

    context = assemble_context(reranked)
    t2 = time.time()
    answer = answer_query(query, context)
    log_event("generate", {"query": query, "latency_s": time.time() - t2})

    return answer, context, reranked

# ----------------------------
# Mini eval
# ----------------------------
EVAL_SET = [
    {"query": "What is hybrid retrieval?", "relevant_doc_ids": {"doc_3"}},
    {"query": "Name chunking strategies.", "relevant_doc_ids": {"doc_2"}},
]

def eval_retrieval():
    hits = 0
    for ex in EVAL_SET:
        ranked_ids, _, _, _ = hybrid_retrieve(ex["query"], chunk_texts, chunk_vectors, topk=3)
        retrieved_docs = {chunks[i]["doc_id"] for i in ranked_ids}
        if len(retrieved_docs & ex["relevant_doc_ids"]) > 0:
            hits += 1
    return hits / len(EVAL_SET)

# ----------------------------
# Run
# ----------------------------
query = "Explain hybrid retrieval and how results are merged."
answer, context, reranked = run_demo(query)

print("Answer:\n", answer)
print("\nContext used:\n", context)
print("\nRetrieval hit@3:", eval_retrieval())
print("\nEvents:", EVENTS)