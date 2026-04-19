"""
Main query script - Search and retrieve from multi-modal RAG system
Usage: python main_query.py --query "What is the total salary?"
"""

import argparse
import logging
from typing import List, Dict, Any
from openai import OpenAI

from config.settings import settings
from ingestion.embedder import embedder
from stores.postgres_store import postgres_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def search_documents(
        query: str,
        top_k: int = 5,
        include_images: bool = True
) -> Dict[str, Any]:
    """
    Search for relevant chunks and images based on query

    Args:
        query: User's question
        top_k: Number of results to return
        include_images: Whether to search images too

    Returns:
        Dictionary with retrieved chunks and media
    """
    logger.info(f"\n🔍 Searching for: '{query}'")

    # Generate query embedding
    logger.info("  - Generating query embedding...")
    query_embedding = embedder.embed_text(query)

    # Search text chunks
    logger.info(f"  - Searching text chunks (top {top_k})...")
    chunks = postgres_store.vector_search(
        query_embedding=query_embedding,
        top_k=top_k
    )

    logger.info(f"  ✓ Found {len(chunks)} relevant chunks")

    # Search images if requested
    media = []
    if include_images:
        logger.info(f"  - Searching images (top 3)...")
        media = postgres_store.vector_search_media(
            query_embedding=query_embedding,
            top_k=3
        )
        logger.info(f"  ✓ Found {len(media)} relevant images")

    return {
        'query': query,
        'chunks': chunks,
        'media': media
    }


def generate_response(
        query: str,
        retrieved_data: Dict[str, Any]
) -> str:
    """
    Generate final answer using retrieved context

    Args:
        query: User's question
        retrieved_data: Retrieved chunks and media

    Returns:
        Generated answer
    """
    logger.info("\n🤖 Generating response...")

    # Build context from chunks
    context_parts = []

    for idx, chunk in enumerate(retrieved_data['chunks'], 1):
        context_parts.append(
            f"[Chunk {idx}] (Page {chunk.get('page_number', 'N/A')}, Similarity: {chunk['similarity']:.3f})\n"
            f"{chunk['text_content']}\n"
        )

    # Add image descriptions if available
    for idx, img in enumerate(retrieved_data['media'], 1):
        if img.get('caption'):
            context_parts.append(
                f"[Image {idx}] (Page {img.get('page_number', 'N/A')}, Similarity: {img['similarity']:.3f})\n"
                f"Description: {img['caption']}\n"
            )

    context = "\n---\n".join(context_parts)

    # Create prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context from document:
{context}

User Question: {query}

Instructions:
- Answer the question based ONLY on the context provided above
- If the answer is not in the context, say "I cannot find this information in the document"
- Be specific and cite which chunk/page the information comes from
- Keep the answer concise and direct

Answer:"""

    # Call OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on provided document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    logger.info("  ✓ Response generated")

    return answer


def display_results(
        query: str,
        retrieved_data: Dict[str, Any],
        answer: str
):
    """Pretty print results"""

    print("\n" + "=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)
    print(f"\n📝 Question: {query}")
    print(f"\n🎯 Answer:\n{answer}")

    print("\n" + "-" * 80)
    print("📚 RETRIEVED CONTEXT")
    print("-" * 80)

    # Show chunks
    print(f"\n✅ Retrieved {len(retrieved_data['chunks'])} text chunks:\n")
    for idx, chunk in enumerate(retrieved_data['chunks'], 1):
        print(f"[{idx}] Chunk ID: {chunk['chunk_id']}")
        print(f"    Page: {chunk.get('page_number', 'N/A')} | Similarity: {chunk['similarity']:.3f}")
        print(f"    Text: {chunk['text_content'][:200]}...")
        print()

    # Show images
    if retrieved_data['media']:
        print(f"\n🖼️  Retrieved {len(retrieved_data['media'])} images:\n")
        for idx, img in enumerate(retrieved_data['media'], 1):
            print(f"[{idx}] Media ID: {img['media_id']}")
            print(f"    Page: {img.get('page_number', 'N/A')} | Similarity: {img['similarity']:.3f}")
            print(f"    File: {img['file_path']}")
            if img.get('caption'):
                print(f"    Description: {img['caption'][:150]}...")
            print()

    print("=" * 80 + "\n")


def interactive_mode():
    """Interactive query mode - keep asking questions"""
    print("\n" + "=" * 80)
    print("🤖 INTERACTIVE RAG QUERY MODE")
    print("=" * 80)
    print("\nType your questions (or 'exit' to quit)\n")

    while True:
        try:
            query = input("\n❓ Your question: ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            # Search
            retrieved_data = search_documents(query, top_k=5, include_images=True)

            # Generate answer
            answer = generate_response(query, retrieved_data)

            # Display
            display_results(query, retrieved_data, answer)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='Query the multi-modal RAG system')
    parser.add_argument(
        '--query',
        type=str,
        help='Question to ask'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of chunks to retrieve (default: 5)'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Exclude images from search'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive mode'
    )

    args = parser.parse_args()

    # Interactive mode
    if args.interactive or not args.query:
        interactive_mode()
        return

    # Single query mode
    try:
        # Search
        retrieved_data = search_documents(
            query=args.query,
            top_k=args.top_k,
            include_images=not args.no_images
        )

        # Generate answer
        answer = generate_response(args.query, retrieved_data)

        # Display results
        display_results(args.query, retrieved_data, answer)

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()