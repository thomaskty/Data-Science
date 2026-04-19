#!/usr/bin/env python3
"""
RAG-Anything Interactive Testing Script

A comprehensive testing script to evaluate RAG-Anything's capabilities with various file formats.
Supports: PDF, Images (JPG, PNG, BMP, TIFF, GIF, WebP), Excel (XLS, XLSX),
         Office docs (DOC, DOCX, PPT, PPTX), Text (TXT, MD)

Usage:
    python test_raganything.py --file document.pdf --api-key YOUR_OPENAI_KEY
    python test_raganything.py --file document.pdf  # Uses OPENAI_API_KEY env var
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional
import dotenv
dotenv.load_dotenv()


# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGTester:
    """Interactive RAG-Anything testing interface"""

    SUPPORTED_FORMATS = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'],
        'office': ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'],
        'text': ['.txt', '.md']
    }

    def __init__(self, api_key: str, base_url: Optional[str] = None, working_dir: str = "./rag_test_storage"):
        self.api_key = api_key
        self.base_url = base_url
        self.working_dir = working_dir
        self.rag = None
        self.current_file = None

    def validate_file(self, file_path: str) -> tuple[bool, str]:
        """Validate if file exists and is supported"""
        file_path = Path(file_path)

        if not file_path.exists():
            return False, f"❌ File not found: {file_path}"

        ext = file_path.suffix.lower()
        all_supported = []
        for formats in self.SUPPORTED_FORMATS.values():
            all_supported.extend(formats)

        if ext not in all_supported:
            return False, f"❌ Unsupported format: {ext}\n   Supported: {', '.join(all_supported)}"

        return True, f"✅ Valid file: {file_path.name} ({ext})"

    def get_file_info(self, file_path: str) -> dict:
        """Get detailed file information"""
        file_path = Path(file_path)
        size_kb = file_path.stat().st_size / 1024

        # Determine file category
        ext = file_path.suffix.lower()
        category = 'unknown'
        for cat, exts in self.SUPPORTED_FORMATS.items():
            if ext in exts:
                category = cat
                break

        return {
            'name': file_path.name,
            'path': str(file_path.absolute()),
            'extension': ext,
            'category': category,
            'size_kb': size_kb,
            'size_mb': size_kb / 1024
        }

    async def initialize_rag(self):
        """Initialize RAG-Anything with OpenAI models"""
        logger.info("🚀 Initializing RAG-Anything...")

        # Create configuration
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True
        )

        # LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs
            )

        # Vision model function for image processing
        def vision_model_func(prompt, system_prompt=None, history_messages=[],
                              image_data=None, messages=None, **kwargs):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o-mini",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                }
                            ]
                        }
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Embedding function
        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=self.api_key,
                base_url=self.base_url
            )
        )

        # Initialize RAG-Anything
        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )

        logger.info("✅ RAG-Anything initialized successfully!")

    async def process_file(self, file_path: str) -> bool:
        """Process a file with RAG-Anything"""
        try:
            info = self.get_file_info(file_path)

            print("\n" + "=" * 70)
            print("📄 FILE PROCESSING")
            print("=" * 70)
            print(f"📁 File: {info['name']}")
            print(f"📂 Path: {info['path']}")
            print(f"📊 Size: {info['size_kb']:.2f} KB ({info['size_mb']:.2f} MB)")
            print(f"🔖 Type: {info['category'].upper()} ({info['extension']})")
            print("=" * 70)

            logger.info(f"🔄 Processing file: {info['name']}")

            # Process document
            await self.rag.process_document_complete(
                file_path=file_path,
                output_dir="./test_output",
                parse_method="auto",
                display_stats=True
            )

            self.current_file = file_path

            print("\n" + "=" * 70)
            print("✅ FILE PROCESSED SUCCESSFULLY!")
            print("=" * 70)
            print("💡 You can now ask questions about the document")
            print("=" * 70 + "\n")

            return True

        except Exception as e:
            logger.error(f"❌ Error processing file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the processed document"""
        if not self.rag or not self.current_file:
            return "❌ Please process a file first before querying"

        try:
            logger.info(f"🔍 Querying: {question}")
            result = await self.rag.aquery(question, mode=mode)
            return result
        except Exception as e:
            logger.error(f"❌ Query error: {str(e)}")
            return f"Error: {str(e)}"

    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 70)
        print("🤖 RAG-ANYTHING INTERACTIVE TESTING SCRIPT")
        print("=" * 70)
        print("Test RAG-Anything's capabilities with various file formats!")
        print("\nSupported formats:")
        for category, extensions in self.SUPPORTED_FORMATS.items():
            print(f"  • {category.upper()}: {', '.join(extensions)}")
        print("=" * 70 + "\n")

    async def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "=" * 70)
        print("💬 INTERACTIVE QUERY MODE")
        print("=" * 70)
        print("Commands:")
        print("  • Type your question and press Enter")
        print("  • 'quit' or 'exit' - Exit the program")
        print("  • 'info' - Show current file information")
        print("  • 'help' - Show this help message")
        print("=" * 70 + "\n")

        while True:
            try:
                question = input("\n❓ Your question: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if question.lower() == 'info':
                    if self.current_file:
                        info = self.get_file_info(self.current_file)
                        print(f"\n📄 Current file: {info['name']}")
                        print(f"   Path: {info['path']}")
                        print(f"   Size: {info['size_kb']:.2f} KB")
                        print(f"   Type: {info['category']} ({info['extension']})")
                    else:
                        print("\n❌ No file processed yet")
                    continue

                if question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  • Type any question about the document")
                    print("  • 'info' - Show file information")
                    print("  • 'quit' or 'exit' - Exit")
                    continue

                print("\n🔄 Processing query...")
                answer = await self.query(question)

                print("\n" + "-" * 70)
                print("💡 ANSWER:")
                print("-" * 70)
                print(answer)
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {str(e)}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="RAG-Anything Interactive Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_raganything.py --file document.pdf
  python test_raganything.py --file report.xlsx --api-key sk-...
  python test_raganything.py --file image.png
        """
    )

    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to the file to process'
    )

    parser.add_argument(
        '--api-key',
        default=os.getenv('OPENAI_API_KEY'),
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )

    parser.add_argument(
        '--base-url',
        default=None,
        help='Optional OpenAI API base URL'
    )

    parser.add_argument(
        '--working-dir',
        default='./rag_test_storage',
        help='Working directory for RAG storage (default: ./rag_test_storage)'
    )

    parser.add_argument(
        '--query', '-q',
        help='Single query mode: ask one question and exit'
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("❌ Error: OpenAI API key is required!")
        print("   Set OPENAI_API_KEY environment variable or use --api-key option")
        sys.exit(1)

    # Create tester instance
    tester = RAGTester(
        api_key=args.api_key,
        base_url=args.base_url,
        working_dir=args.working_dir
    )

    # Print welcome message
    tester.print_welcome()

    # Validate file
    is_valid, message = tester.validate_file(args.file)
    print(message)

    if not is_valid:
        sys.exit(1)

    # Initialize RAG
    await tester.initialize_rag()

    # Process file
    success = await tester.process_file(args.file)

    if not success:
        print("\n❌ Failed to process file. Exiting.")
        sys.exit(1)

    # Query mode
    if args.query:
        # Single query mode
        print(f"\n❓ Question: {args.query}")
        answer = await tester.query(args.query)
        print("\n💡 Answer:")
        print(answer)
    else:
        # Interactive mode
        await tester.interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)