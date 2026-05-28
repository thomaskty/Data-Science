from typing import List, Dict, Any
import re
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chunker:
    """
    Handles different chunking strategies for text content
    """

    def __init__(
            self,
            chunk_size: int = None,
            chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(f"✓ Chunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk_text_fixed(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        Simple fixed-size chunking with overlap

        Args:
            text: Input text to chunk
            metadata: Additional metadata to attach to chunks

        Returns:
            List of chunk dictionaries
        """
        # Simple token approximation (1 token ≈ 4 characters)
        chars_per_chunk = self.chunk_size * 4
        chars_overlap = self.chunk_overlap * 4

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chars_per_chunk
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chars_per_chunk * 0.5:  # At least 50% of chunk
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1

            chunk = {
                'text': chunk_text.strip(),
                'chunk_index': chunk_index,
                'start_char': start,
                'end_char': end,
                'token_count': len(chunk_text) // 4,  # Approximation
                'metadata': metadata or {}
            }

            chunks.append(chunk)

            # Move start position with overlap
            start = end - chars_overlap
            chunk_index += 1

        logger.info(f"✓ Created {len(chunks)} chunks using fixed-size strategy")
        return chunks

    def chunk_text_semantic(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        Semantic chunking - splits by paragraphs and sentences
        Better preserves meaning

        Args:
            text: Input text to chunk
            metadata: Additional metadata to attach to chunks

        Returns:
            List of chunk dictionaries
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size * 4:
                if current_chunk:
                    chunk = {
                        'text': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'start_char': start_char,
                        'end_char': start_char + len(current_chunk),
                        'token_count': len(current_chunk) // 4,
                        'metadata': metadata or {}
                    }
                    chunks.append(chunk)
                    chunk_index += 1
                    start_char += len(current_chunk)
                    current_chunk = ""

            current_chunk += para + "\n\n"

        # Add last chunk
        if current_chunk:
            chunk = {
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'start_char': start_char,
                'end_char': start_char + len(current_chunk),
                'token_count': len(current_chunk) // 4,
                'metadata': metadata or {}
            }
            chunks.append(chunk)

        logger.info(f"✓ Created {len(chunks)} chunks using semantic strategy")
        return chunks

    def chunk_text_recursive(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        Recursive chunking - tries multiple separators in order
        Most sophisticated approach

        Hierarchy: \n\n -> \n -> . -> space

        Args:
            text: Input text to chunk
            metadata: Additional metadata to attach to chunks

        Returns:
            List of chunk dictionaries
        """
        separators = ["\n\n", "\n", ". ", " "]

        def _split_recursive(text: str, separators: List[str], max_size: int) -> List[str]:
            """Recursively split text using different separators"""
            if len(text) <= max_size:
                return [text]

            if not separators:
                # Fallback to character split
                return [text[i:i + max_size] for i in range(0, len(text), max_size)]

            separator = separators[0]
            remaining_separators = separators[1:]

            splits = text.split(separator)
            result = []
            current_chunk = ""

            for split in splits:
                if len(current_chunk) + len(split) + len(separator) <= max_size:
                    current_chunk += split + separator
                else:
                    if current_chunk:
                        result.append(current_chunk.strip())

                    # If single split is too large, recurse with next separator
                    if len(split) > max_size:
                        result.extend(_split_recursive(split, remaining_separators, max_size))
                    else:
                        current_chunk = split + separator

            if current_chunk:
                result.append(current_chunk.strip())

            return result

        # Split text
        text_chunks = _split_recursive(text, separators, self.chunk_size * 4)

        # Convert to chunk dictionaries
        chunks = []
        start_char = 0

        for idx, chunk_text in enumerate(text_chunks):
            chunk = {
                'text': chunk_text.strip(),
                'chunk_index': idx,
                'start_char': start_char,
                'end_char': start_char + len(chunk_text),
                'token_count': len(chunk_text) // 4,
                'metadata': metadata or {}
            }
            chunks.append(chunk)
            start_char += len(chunk_text)

        logger.info(f"✓ Created {len(chunks)} chunks using recursive strategy")
        return chunks

    def chunk_by_page(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text by page (for PDFs)
        Preserves page boundaries

        Args:
            pages: List of page dictionaries with 'page_number' and 'text'

        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        chunk_index = 0

        for page in pages:
            page_text = page.get('text', '').strip()
            if not page_text:
                continue

            # Chunk the page text
            page_chunks = self.chunk_text_recursive(
                page_text,
                metadata={'page_number': page.get('page_number')}
            )

            # Update chunk indices and add page number
            for chunk in page_chunks:
                chunk['chunk_index'] = chunk_index
                chunk['page_number'] = page.get('page_number')
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"✓ Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks


# Create singleton instance
chunker = Chunker()