-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- 1. DOCUMENTS TABLE (Raw document storage)
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_path TEXT,
    file_type VARCHAR(50),
    total_pages INTEGER,
    file_size_bytes BIGINT,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    processing_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT
);

CREATE INDEX idx_documents_id ON documents(document_id);
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

-- ============================================
-- 2. CHUNKS TABLE (Text chunks with vectors)
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    token_count INTEGER,
    page_number INTEGER,

    -- Vector embeddings
    embedding vector(1536),  -- Adjust dimension based on your model

    -- Chunk metadata
    chunk_type VARCHAR(50) DEFAULT 'text',  -- 'text', 'heading', 'list', 'code'
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_id ON chunks(chunk_id);
CREATE INDEX idx_chunks_page ON chunks(page_number);
CREATE INDEX idx_chunks_metadata ON chunks USING GIN(metadata);

-- Vector similarity search index (HNSW is faster than IVFFlat for most cases)
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- ============================================
-- 3. MEDIA TABLE (Images, tables, diagrams)
-- ============================================
CREATE TABLE IF NOT EXISTS media (
    id SERIAL PRIMARY KEY,
    media_id VARCHAR(255) UNIQUE NOT NULL,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    media_type VARCHAR(50) NOT NULL,  -- 'image', 'table', 'chart', 'diagram'
    page_number INTEGER,

    -- Storage
    file_path TEXT NOT NULL,
    thumbnail_path TEXT,

    -- Embeddings (CLIP for images)
    embedding vector(1536),  -- CLIP ViT-B/32 dimension

    -- Media metadata
    width INTEGER,
    height INTEGER,
    caption TEXT,
    extracted_text TEXT,  -- OCR or table text
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_media_document ON media(document_id);
CREATE INDEX idx_media_id ON media(media_id);
CREATE INDEX idx_media_type ON media(media_type);
CREATE INDEX idx_media_page ON media(page_number);
CREATE INDEX idx_media_embedding ON media USING hnsw (embedding vector_cosine_ops);

-- ============================================
-- 4. METADATA TABLE (Structured metadata & filters)
-- ============================================
CREATE TABLE IF NOT EXISTS document_metadata (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,

    -- Common metadata fields
    title TEXT,
    author VARCHAR(500),
    created_date DATE,
    modified_date DATE,
    source VARCHAR(500),
    category VARCHAR(100),
    tags TEXT[],

    -- Custom metadata
    custom_fields JSONB DEFAULT '{}',

    -- Search optimization
    searchable_text TEXT,  -- Concatenated metadata for full-text search

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metadata_document ON document_metadata(document_id);
CREATE INDEX idx_metadata_category ON document_metadata(category);
CREATE INDEX idx_metadata_tags ON document_metadata USING GIN(tags);
CREATE INDEX idx_metadata_search ON document_metadata USING GIN(to_tsvector('english', searchable_text));

-- ============================================
-- 5. RELATIONSHIPS TABLE (Graph edges)
-- ============================================
CREATE TABLE IF NOT EXISTS relationships (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- 'document', 'chunk', 'media'
    target_id VARCHAR(255) NOT NULL,
    target_type VARCHAR(50) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,  -- 'contains', 'references', 'similar_to', 'follows'

    -- Relationship metadata
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_id, target_id, relationship_type)
);

CREATE INDEX idx_relationships_source ON relationships(source_id, source_type);
CREATE INDEX idx_relationships_target ON relationships(target_id, target_type);
CREATE INDEX idx_relationships_type ON relationships(relationship_type);

-- ============================================
-- 6. QUERY LOGS (Optional - for analytics)
-- ============================================
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(1536),

    -- Results
    top_chunks TEXT[],
    top_media TEXT[],
    retrieval_method VARCHAR(50),  -- 'vector', 'bm25', 'hybrid'

    -- Performance metrics
    retrieval_time_ms INTEGER,
    rerank_time_ms INTEGER,
    total_time_ms INTEGER,

    -- User feedback (optional)
    user_rating INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_query_logs_created ON query_logs(created_at);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to calculate cosine similarity
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get similar chunks
CREATE OR REPLACE FUNCTION get_similar_chunks(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id VARCHAR,
    text_content TEXT,
    similarity FLOAT,
    document_id VARCHAR,
    page_number INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        c.text_content,
        1 - (c.embedding <=> query_embedding) AS similarity,
        c.document_id,
        c.page_number
    FROM chunks c
    WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- VIEWS FOR CONVENIENCE
-- ============================================

-- View combining chunks with document info
CREATE OR REPLACE VIEW chunks_with_document AS
SELECT
    c.*,
    d.filename,
    d.file_type,
    d.metadata as document_metadata
FROM chunks c
JOIN documents d ON c.document_id = d.document_id;

-- View combining media with document info
CREATE OR REPLACE VIEW media_with_document AS
SELECT
    m.*,
    d.filename,
    d.file_type
FROM media m
JOIN documents d ON m.document_id = d.document_id;