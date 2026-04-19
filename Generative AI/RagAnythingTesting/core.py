import asyncio
import os
from pathlib import Path  # ADD THIS IMPORT

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import dotenv

dotenv.load_dotenv()

API_KEY = os.getenv('API_KEY')
WORKING_DIR = os.getenv('WORKING_DIR')
LLM_MODEL = os.getenv('LLM_MODEL')
VISION_MODEL = os.getenv('VISION_MODEL')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1536'))  # Convert to int
PARSER = os.getenv('PARSER')
PARSE_METHOD = os.getenv('PARSE_METHOD')
ENABLE_IMAGE = os.getenv('ENABLE_IMAGE', 'True').lower() == 'true'  # Convert to bool
ENABLE_TABLE = os.getenv('ENABLE_TABLE', 'True').lower() == 'true'
ENABLE_EQUATION = os.getenv('ENABLE_EQUATION', 'True').lower() == 'true'
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1200'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))


class SimpleRAG:
    """Minimal RAG-Anything wrapper - configured via .env"""

    def __init__(self, api_key=None):
        self.api_key = api_key or API_KEY
        self.rag = None

    async def initialize(self):
        """Initialize RAG-Anything"""
        config = RAGAnythingConfig(
            working_dir=WORKING_DIR,
            parser=PARSER,
            parse_method=PARSE_METHOD,
            enable_image_processing=ENABLE_IMAGE,
            enable_table_processing=ENABLE_TABLE,
            enable_equation_processing=ENABLE_EQUATION
        )

        def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.setdefault('max_tokens', MAX_TOKENS)
            return openai_complete_if_cache(
                LLM_MODEL, prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=self.api_key, **kwargs
            )

        def vision_func(prompt, system_prompt=None, history_messages=[],
                        image_data=None, messages=None, **kwargs):
            kwargs.setdefault('max_tokens', MAX_TOKENS)

            if messages:
                return openai_complete_if_cache(
                    VISION_MODEL, "", messages=messages,
                    api_key=self.api_key, **kwargs
                )
            elif image_data:
                return openai_complete_if_cache(
                    VISION_MODEL, "",
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                            ]
                        }
                    ],
                    api_key=self.api_key, **kwargs
                )
            else:
                return llm_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, model=EMBEDDING_MODEL, api_key=self.api_key
            )
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embedding_func
        )

        # ⬇️⬇️⬇️ ADD THESE 3 LINES HERE ⬇️⬇️⬇️
        # Load existing storage if available
        if Path(WORKING_DIR).exists():
            await self.rag._ensure_lightrag_initialized()

    async def process(self, file_path: str):
        """Process document"""
        await self.rag.process_document_complete(
            file_path=file_path,
            output_dir="./output",
            parse_method=PARSE_METHOD,
            display_stats=False
        )

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the document"""
        return await self.rag.aquery(question, mode=mode)


async def main():
    rag = SimpleRAG()
    await rag.initialize()

    # STOP : IF FILE IS ALREADY PROCESSED , COMMENT FOLLOWING LINE
    # await rag.process("research_paper.pdf")

    answer = await rag.query("give the statistics of experimental datasets table as a json")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())