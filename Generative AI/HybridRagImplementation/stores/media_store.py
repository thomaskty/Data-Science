from pathlib import Path
from typing import List, Dict, Any
import shutil
import logging
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaStore:
    """
    Manages file system storage for media (images, tables, etc.)
    Works alongside PostgreSQL media table
    """

    def __init__(self):
        self.storage_dir = settings.MEDIA_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Media store initialized: {self.storage_dir}")

    def store_media_file(self, source_path: str, media_id: str, media_type: str) -> str:
        """
        Copy media file to storage directory

        Args:
            source_path: Original file path
            media_id: Unique media identifier
            media_type: Type of media (image, table, etc.)

        Returns:
            New file path in storage
        """
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Create subdirectory by type
        type_dir = self.storage_dir / media_type
        type_dir.mkdir(exist_ok=True)

        # New filename with media_id
        extension = source.suffix
        new_filename = f"{media_id}{extension}"
        dest_path = type_dir / new_filename

        # Copy file
        shutil.copy2(source, dest_path)

        logger.info(f"✓ Stored media: {new_filename}")
        return str(dest_path)

    def get_media_path(self, media_id: str, media_type: str) -> str:
        """
        Get the file path for a media item

        Args:
            media_id: Media identifier
            media_type: Type of media

        Returns:
            File path
        """
        # Try to find file in type directory
        type_dir = self.storage_dir / media_type

        if not type_dir.exists():
            raise FileNotFoundError(f"Media type directory not found: {media_type}")

        # Find file with media_id (any extension)
        for file in type_dir.glob(f"{media_id}.*"):
            return str(file)

        raise FileNotFoundError(f"Media file not found: {media_id}")

    def delete_media(self, media_id: str, media_type: str):
        """Delete a media file"""
        try:
            file_path = self.get_media_path(media_id, media_type)
            Path(file_path).unlink()
            logger.info(f"✓ Deleted media: {media_id}")
        except FileNotFoundError:
            logger.warning(f"Media file not found for deletion: {media_id}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_type': {}
        }

        for type_dir in self.storage_dir.iterdir():
            if type_dir.is_dir():
                files = list(type_dir.glob('*'))
                file_count = len(files)
                total_size = sum(f.stat().st_size for f in files if f.is_file())

                stats['by_type'][type_dir.name] = {
                    'count': file_count,
                    'size_bytes': total_size
                }

                stats['total_files'] += file_count
                stats['total_size_bytes'] += total_size

        return stats


# Create singleton instance
media_store = MediaStore()