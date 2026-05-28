import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgresStore:
    """
    Handles all PostgreSQL operations for the multi-modal RAG system
    """

    def __init__(self):
        self.connection_params = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'database': settings.POSTGRES_DB,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD
        }
        self._test_connection()

    def _test_connection(self):
        """Test database connection"""
        try:
            conn = self.get_connection()
            conn.close()
            logger.info("✓ Database connection successful")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise

    def get_connection(self):
        """Get a new database connection"""
        return psycopg2.connect(**self.connection_params)

    def initialize_schema(self, schema_path: str):
        """Execute schema SQL file to create tables"""
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(schema_sql)
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✓ Database schema initialized")
        except Exception as e:
            logger.error(f"✗ Schema initialization failed: {e}")
            raise

    # ==================== DOCUMENT OPERATIONS ====================

    def insert_document(self, document_data: Dict[str, Any]) -> str:
        """Insert a new document record"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO documents (
            document_id, filename, file_path, file_type, 
            total_pages, file_size_bytes, metadata, processing_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING document_id
        """

        cursor.execute(query, (
            document_data['document_id'],
            document_data['filename'],
            document_data.get('file_path'),
            document_data.get('file_type'),
            document_data.get('total_pages'),
            document_data.get('file_size_bytes'),
            psycopg2.extras.Json(document_data.get('metadata', {})),
            document_data.get('processing_status', 'pending')
        ))

        document_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✓ Document inserted: {document_id}")
        return document_id

    def update_document_status(self, document_id: str, status: str, error_message: Optional[str] = None):
        """Update document processing status"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
        UPDATE documents 
        SET processing_status = %s, error_message = %s
        WHERE document_id = %s
        """

        cursor.execute(query, (status, error_message, document_id))
        conn.commit()
        cursor.close()
        conn.close()

    # ==================== CHUNK OPERATIONS ====================

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Batch insert text chunks with embeddings"""
        if not chunks:
            return

        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO chunks (
            chunk_id, document_id, chunk_index, text_content, 
            token_count, page_number, embedding, chunk_type, metadata
        ) VALUES %s
        """

        values = [
            (
                chunk['chunk_id'],
                chunk['document_id'],
                chunk['chunk_index'],
                chunk['text_content'],
                chunk.get('token_count'),
                chunk.get('page_number'),
                chunk['embedding'],  # Already as list
                chunk.get('chunk_type', 'text'),
                psycopg2.extras.Json(chunk.get('metadata', {}))
            )
            for chunk in chunks
        ]

        execute_values(cursor, query, values)
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✓ Inserted {len(chunks)} chunks")

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT chunk_id, text_content, page_number, chunk_type, metadata
        FROM chunks
        WHERE document_id = %s
        ORDER BY chunk_index
        """

        cursor.execute(query, (document_id,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return [dict(row) for row in results]

    # ==================== VECTOR SEARCH OPERATIONS ====================

    def vector_search(
            self,
            query_embedding: List[float],
            top_k: int = 10,
            document_id: Optional[str] = None,
            metadata_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Build query with optional filters
        query = """
        SELECT 
            chunk_id,
            text_content,
            document_id,
            page_number,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        WHERE 1=1
        """

        params = [query_embedding]

        if document_id:
            query += " AND document_id = %s"
            params.append(document_id)

        if metadata_filter:
            # Simple metadata filtering (can be expanded)
            for key, value in metadata_filter.items():
                query += f" AND metadata->>%s = %s"
                params.extend([key, str(value)])

        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, top_k])

        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return [dict(row) for row in results]

    # ==================== MEDIA OPERATIONS ====================

    def insert_media_batch(self, media_items: List[Dict[str, Any]]):
        """Batch insert media items with embeddings"""
        if not media_items:
            return

        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO media (
            media_id, document_id, media_type, page_number,
            file_path, thumbnail_path, embedding, width, height,
            caption, extracted_text, metadata
        ) VALUES %s
        """

        values = [
            (
                item['media_id'],
                item['document_id'],
                item['media_type'],
                item.get('page_number'),
                item['file_path'],
                item.get('thumbnail_path'),
                item.get('embedding'),
                item.get('width'),
                item.get('height'),
                item.get('caption'),
                item.get('extracted_text'),
                psycopg2.extras.Json(item.get('metadata', {}))
            )
            for item in media_items
        ]

        execute_values(cursor, query, values)
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✓ Inserted {len(media_items)} media items")

    def vector_search_media(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            media_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search media by vector similarity"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT 
            media_id,
            media_type,
            file_path,
            page_number,
            caption,
            extracted_text,
            document_id,
            1 - (embedding <=> %s::vector) AS similarity
        FROM media
        WHERE 1=1
        """

        params = [query_embedding]

        if media_type:
            query += " AND media_type = %s"
            params.append(media_type)

        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, top_k])

        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return [dict(row) for row in results]

    # ==================== METADATA OPERATIONS ====================

    def insert_metadata(self, metadata: Dict[str, Any]):
        """Insert document metadata"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO document_metadata (
            document_id, title, author, created_date, source,
            category, tags, custom_fields, searchable_text
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        searchable = f"{metadata.get('title', '')} {metadata.get('author', '')} {' '.join(metadata.get('tags', []))}"

        cursor.execute(query, (
            metadata['document_id'],
            metadata.get('title'),
            metadata.get('author'),
            metadata.get('created_date'),
            metadata.get('source'),
            metadata.get('category'),
            metadata.get('tags', []),
            psycopg2.extras.Json(metadata.get('custom_fields', {})),
            searchable
        ))

        conn.commit()
        cursor.close()
        conn.close()

    # ==================== RELATIONSHIP OPERATIONS ====================

    def insert_relationships_batch(self, relationships: List[Dict[str, Any]]):
        """Batch insert graph relationships"""
        if not relationships:
            return

        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO relationships (
            source_id, source_type, target_id, target_type,
            relationship_type, weight, metadata
        ) VALUES %s
        ON CONFLICT (source_id, target_id, relationship_type) DO NOTHING
        """

        values = [
            (
                rel['source_id'],
                rel['source_type'],
                rel['target_id'],
                rel['target_type'],
                rel['relationship_type'],
                rel.get('weight', 1.0),
                psycopg2.extras.Json(rel.get('metadata', {}))
            )
            for rel in relationships
        ]

        execute_values(cursor, query, values)
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✓ Inserted {len(relationships)} relationships")

    # ==================== UTILITY OPERATIONS ====================

    def get_document_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT 
            (SELECT COUNT(*) FROM documents) as total_documents,
            (SELECT COUNT(*) FROM chunks) as total_chunks,
            (SELECT COUNT(*) FROM media) as total_media,
            (SELECT COUNT(*) FROM relationships) as total_relationships
        """

        cursor.execute(query)
        stats = dict(cursor.fetchone())
        cursor.close()
        conn.close()

        return stats

    def clear_all_data(self):
        """Clear all data from tables (for testing)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        tables = ['relationships', 'media', 'chunks', 'document_metadata', 'documents', 'query_logs']

        for table in tables:
            cursor.execute(f"TRUNCATE TABLE {table} CASCADE")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("✓ All data cleared")


# Create singleton instance
postgres_store = PostgresStore()