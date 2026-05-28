from openai import OpenAI
from typing import List, Union
import numpy as np
from config.settings import settings
import logging
from PIL import Image
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles text and image embeddings using OpenAI models
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.text_model = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        logger.info(f"✓ Embedder initialized with model: {self.text_model}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text string

        Returns:
            List of floats representing the embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.text_model,
                input=text,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise

    def embed_texts_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of text strings
            batch_size: Number of texts to process per API call

        Returns:
            List of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.text_model,
                    input=batch,
                    dimensions=self.dimensions
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.info(
                    f"✓ Generated embeddings for batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise

        return all_embeddings

    def embed_image_clip(self, image_path: str) -> List[float]:
        """
        Generate CLIP embedding for an image using OpenAI's vision model
        Note: OpenAI doesn't expose CLIP embeddings directly, so we'll use a workaround
        or you can integrate with open-source CLIP model

        For now, this is a placeholder for CLIP embeddings
        In production, you'd use: from sentence_transformers import SentenceTransformer
        and load 'clip-ViT-B-32' model

        Args:
            image_path: Path to image file

        Returns:
            List of floats representing the embedding
        """
        # TODO: Implement actual CLIP embeddings
        # For now, return a zero vector as placeholder
        logger.warning("CLIP embeddings not yet implemented - returning placeholder")
        return [0.0] * 512  # CLIP ViT-B/32 has 512 dimensions

    def describe_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Use GPT-4 Vision to generate text description of an image

        Args:
            image_path: Path to image file
            prompt: Custom prompt for image description

        Returns:
            Text description of the image
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            base64_image = base64.b64encode(image_data).decode('utf-8')

            response = self.client.chat.completions.create(
                model=settings.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            description = response.choices[0].message.content
            logger.info(f"✓ Generated image description")
            return description

        except Exception as e:
            logger.error(f"Error describing image: {e}")
            raise

    def embed_image_via_description(self, image_path: str) -> List[float]:
        """
        Generate text embedding for an image by first describing it with GPT-4 Vision
        This allows us to use text embeddings for images

        Args:
            image_path: Path to image file

        Returns:
            List of floats representing the embedding
        """
        description = self.describe_image(image_path)
        return self.embed_text(description)


# Create singleton instance
embedder = Embedder()