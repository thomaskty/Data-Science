"""
Main ingestion script - Process PDF and store in all stores
Usage: python main_ingest.py --pdf sample_data/your_file.pdf
"""

import argparse
import uuid
from pathlib import Path
import logging
from tqdm import tqdm

from config.settings import settings
from ingestion.parser import pdf_parser
from ingestion.chunker import chunker
from ingestion.embedder import embedder
from stores.postgres_store import postgres_store
from stores.media_store import media_store
from stores.graph_store import graph_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_id(prefix: str) -> str:
    """Generate unique ID with prefix"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def ingest_pdf(pdf_path: str, chunking_strategy: str = 'recursive'):
    """
    Main ingestion pipeline

    Steps:
    1. Parse PDF (extract text, images, metadata)
    2. Chunk text
    3. Generate embeddings
    4. Store in Postgres (documents, chunks, media)
    5. Build graph

    Args:
        pdf_path: Path to PDF file
        chunking_strategy: 'fixed', 'semantic', or 'recursive'
    """

    logger.info("=" * 60)
    logger.info("STARTING INGESTION PIPELINE")
    logger.info("=" * 60)

    # Generate document ID
    document_id = generate_id('doc')

    # ==================== STEP 1: Parse PDF ====================
    logger.info("\n[STEP 1/5] Parsing PDF...")
    parsed_data = pdf_parser.parse_pdf(pdf_path)

    # ==================== STEP 2: Store Document ====================
    logger.info("\n[STEP 2/5] Storing document metadata...")
    document_data = {
        'document_id': document_id,
        'filename': parsed_data['filename'],
        'file_path': parsed_data['file_path'],
        'file_type': 'pdf',
        'total_pages': parsed_data['total_pages'],
        'file_size_bytes': parsed_data['file_size_bytes'],
        'metadata': parsed_data['metadata'],
        'processing_status': 'processing'
    }

    postgres_store.insert_document(document_data)

    # ==================== STEP 3: Chunk Text ====================
    logger.info(f"\n[STEP 3/5] Chunking text using '{chunking_strategy}' strategy...")

    # Choose chunking strategy
    if chunking_strategy == 'fixed':
        all_chunks = chunker.chunk_text_fixed(
            '\n\n'.join([p['text'] for p in parsed_data['pages']])
        )
    elif chunking_strategy == 'semantic':
        all_chunks = chunker.chunk_text_semantic(
            '\n\n'.join([p['text'] for p in parsed_data['pages']])
        )
    else:  # recursive (default)
        all_chunks = chunker.chunk_by_page(parsed_data['pages'])

    logger.info(f"✓ Created {len(all_chunks)} chunks")

    # ==================== STEP 4: Generate Embeddings ====================
    logger.info("\n[STEP 4/5] Generating embeddings...")

    # Extract texts for embedding
    chunk_texts = [chunk['text'] for chunk in all_chunks]

    logger.info("  - Generating text embeddings (this may take a while)...")
    embeddings = embedder.embed_texts_batch(chunk_texts, batch_size=50)

    # Prepare chunks for database
    chunks_for_db = []
    for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        chunk_id = generate_id('chunk')
        chunks_for_db.append({
            'chunk_id': chunk_id,
            'document_id': document_id,
            'chunk_index': idx,
            'text_content': chunk['text'],
            'token_count': chunk['token_count'],
            'page_number': chunk.get('page_number'),
            'embedding': embedding,
            'chunk_type': 'text',
            'metadata': chunk.get('metadata', {})
        })

    # Store chunks in Postgres
    postgres_store.insert_chunks_batch(chunks_for_db)

    # ==================== STEP 5: Process Images ====================
    logger.info("\n[STEP 5/5] Processing images...")

    media_items = []

    if parsed_data['images']:
        logger.info(f"  - Processing {len(parsed_data['images'])} images...")

        for img in tqdm(parsed_data['images'], desc="Processing images"):
            media_id = generate_id('media')

            try:
                # Generate image description using GPT-4 Vision
                logger.info(f"  - Describing image: {img['filename']}")
                description = embedder.describe_image(img['file_path'])

                # Generate embedding from description
                img_embedding = embedder.embed_text(description)

                media_item = {
                    'media_id': media_id,
                    'document_id': document_id,
                    'media_type': 'image',
                    'page_number': img['page_number'],
                    'file_path': img['file_path'],
                    'thumbnail_path': None,
                    'embedding': img_embedding,
                    'width': img['width'],
                    'height': img['height'],
                    'caption': description,
                    'extracted_text': None,
                    'metadata': {
                        'format': img['format'],
                        'size_bytes': img['size_bytes']
                    }
                }

                media_items.append(media_item)

            except Exception as e:
                logger.error(f"  ✗ Failed to process image {img['filename']}: {e}")
                continue

        # Store media in Postgres
        if media_items:
            postgres_store.insert_media_batch(media_items)
    else:
        logger.info("  - No images found in PDF")

    # ==================== STEP 6: Build Graph ====================
    logger.info("\n[STEP 6/5] Building knowledge graph...")

    if settings.BUILD_GRAPH:
        graph_store.build_document_graph(
            document_id=document_id,
            chunks=chunks_for_db,
            media=media_items
        )
        graph_store.save_graph()

    # ==================== STEP 7: Build Relationships ====================
    logger.info("\n[STEP 7/5] Creating relationships...")

    relationships = []

    # Document -> Chunk relationships
    for chunk in chunks_for_db:
        relationships.append({
            'source_id': document_id,
            'source_type': 'document',
            'target_id': chunk['chunk_id'],
            'target_type': 'chunk',
            'relationship_type': 'contains',
            'weight': 1.0
        })

    # Document -> Media relationships
    for media_item in media_items:
        relationships.append({
            'source_id': document_id,
            'source_type': 'document',
            'target_id': media_item['media_id'],
            'target_type': 'media',
            'relationship_type': 'contains',
            'weight': 1.0
        })

    # Chunk -> Media relationships (same page)
    for media_item in media_items:
        page_num = media_item.get('page_number')
        if page_num:
            for chunk in chunks_for_db:
                if chunk.get('page_number') == page_num:
                    relationships.append({
                        'source_id': chunk['chunk_id'],
                        'source_type': 'chunk',
                        'target_id': media_item['media_id'],
                        'target_type': 'media',
                        'relationship_type': 'references',
                        'weight': 0.8
                    })

    postgres_store.insert_relationships_batch(relationships)

    # ==================== Update Status ====================
    postgres_store.update_document_status(document_id, 'completed')

    # ==================== Summary ====================
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Document ID: {document_id}")
    logger.info(f"Chunks created: {len(chunks_for_db)}")
    logger.info(f"Images processed: {len(media_items)}")
    logger.info(f"Relationships: {len(relationships)}")

    # Database stats
    stats = postgres_store.get_document_stats()
    logger.info("\nDatabase Statistics:")
    logger.info(f"  Total documents: {stats['total_documents']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Total media: {stats['total_media']}")
    logger.info(f"  Total relationships: {stats['total_relationships']}")

    if settings.BUILD_GRAPH:
        graph_stats = graph_store.get_stats()
        logger.info("\nGraph Statistics:")
        logger.info(f"  Total nodes: {graph_stats['total_nodes']}")
        logger.info(f"  Total edges: {graph_stats['total_edges']}")
        logger.info(f"  Node types: {graph_stats['node_types']}")

    logger.info("=" * 60)

    return document_id


def main():
    parser = argparse.ArgumentParser(description='Ingest PDF into multi-modal RAG system')
    parser.add_argument(
        '--pdf',
        type=str,
        required=True,
        help='Path to PDF file'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='recursive',
        choices=['fixed', 'semantic', 'recursive'],
        help='Chunking strategy'
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database schema before ingestion'
    )

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database schema...")
        schema_path = Path(__file__).parent / 'storage' / 'schema.sql'
        postgres_store.initialize_schema(str(schema_path))
        logger.info("✓ Database initialized")

    # Check if PDF exists
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    # Run ingestion
    try:
        document_id = ingest_pdf(str(pdf_path), args.strategy)
        logger.info(f"\n✓ SUCCESS! Document ID: {document_id}")
    except Exception as e:
        logger.error(f"\n✗ INGESTION FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    main()