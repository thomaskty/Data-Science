import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from PIL import Image
import io
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parses PDF documents to extract text, images, tables, and metadata
    Uses PyMuPDF (fitz) for robust PDF processing
    """

    def __init__(self):
        self.media_dir = settings.MEDIA_STORAGE_DIR
        self.media_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✓ PDF Parser initialized")

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main parsing function - extracts everything from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing:
                - metadata: PDF metadata
                - pages: List of page dictionaries with text
                - images: List of extracted images with metadata
                - tables: List of detected tables
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"📄 Parsing PDF: {pdf_path.name}")

        doc = fitz.open(pdf_path)

        result = {
            'filename': pdf_path.name,
            'file_path': str(pdf_path.absolute()),
            'total_pages': len(doc),
            'file_size_bytes': pdf_path.stat().st_size,
            'metadata': self._extract_metadata(doc),
            'pages': [],
            'images': [],
            'tables': []
        }

        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            page_text = page.get_text("text")

            page_data = {
                'page_number': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            }

            result['pages'].append(page_data)

            # Extract images from this page
            page_images = self._extract_images_from_page(
                page,
                page_num + 1,
                pdf_path.stem
            )
            result['images'].extend(page_images)

            logger.info(f"✓ Processed page {page_num + 1}/{len(doc)}")

        doc.close()

        logger.info(f"✓ Extraction complete: {len(result['pages'])} pages, {len(result['images'])} images")
        return result

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = doc.metadata or {}

        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'mod_date': metadata.get('modDate', '')
        }

    def _extract_images_from_page(
            self,
            page: fitz.Page,
            page_number: int,
            doc_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF page

        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            doc_name: Document name for file naming

        Returns:
            List of image dictionaries
        """
        images = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                # Extract image
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Open with PIL
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Save image
                image_filename = f"{doc_name}_p{page_number}_img{img_index + 1}.{image_ext}"
                image_path = self.media_dir / image_filename

                # Resize if too large
                if pil_image.width > settings.MAX_IMAGE_SIZE[0] or pil_image.height > settings.MAX_IMAGE_SIZE[1]:
                    pil_image.thumbnail(settings.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

                pil_image.save(image_path, quality=settings.IMAGE_QUALITY)

                image_data = {
                    'page_number': page_number,
                    'image_index': img_index + 1,
                    'file_path': str(image_path),
                    'filename': image_filename,
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'format': image_ext,
                    'size_bytes': len(image_bytes)
                }

                images.append(image_data)
                logger.info(f"  ✓ Extracted image: {image_filename}")

            except Exception as e:
                logger.warning(f"  ✗ Failed to extract image {img_index + 1} from page {page_number}: {e}")
                continue

        return images

    def extract_text_only(self, pdf_path: str) -> str:
        """
        Quick extraction of all text from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            All text content concatenated
        """
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text("text") + "\n\n"

        doc.close()
        return text.strip()

    def get_page_text(self, pdf_path: str, page_number: int) -> str:
        """
        Extract text from a specific page

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            Text content of the page
        """
        doc = fitz.open(pdf_path)

        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Invalid page number: {page_number}")

        page = doc[page_number - 1]
        text = page.get_text("text")
        doc.close()

        return text


# Create singleton instance
pdf_parser = PDFParser()