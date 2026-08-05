import os
import warnings
# Suppress LangChain deprecation warnings to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging

logger = logging.getLogger("DataAnalystAgent.Knowledge")

# Configuration
default_chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", default_chroma_dir)
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

class MockEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.0] * 384

class KnowledgeBase:
    def __init__(self):
        self.n_results = 3 # Default depth
        self.embeddings = None
        self.vector_store = None
        self.init_error = None
        self.using_fallback = False

        # Initialize eagerly when possible, but never block app startup.
        self._ensure_ready()

    def _create_vector_store(self, persist_directory=None):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def _ensure_ready(self):
        if self.vector_store is not None:
            return True

        if os.getenv("TESTING") == "true":
            try:
                self.embeddings = MockEmbeddings()
                self.vector_store = self._create_vector_store()
                self.init_error = None
                self.using_fallback = True
                logger.info("[Test] Knowledge Base using in-memory mock store.")
                return True
            except Exception as e:
                self.init_error = str(e)
                self.vector_store = None
                self.using_fallback = False
                return False

        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
            self.embeddings = OllamaEmbeddings(
                base_url=base_url,
                model=EMBEDDING_MODEL
            )
            self.vector_store = self._create_vector_store(PERSIST_DIRECTORY)
            self.init_error = None
            self.using_fallback = False
            return True
        except Exception as e:
            persistent_error = str(e)
            logger.warning(f"Persistent Knowledge Base unavailable: {persistent_error}")

        try:
            self.vector_store = self._create_vector_store()
            self.init_error = None
            self.using_fallback = True
            logger.info("Knowledge Base using in-memory fallback store.")
            return True
        except Exception as e:
            self.init_error = str(e)
            self.vector_store = None
            self.using_fallback = False
            logger.error(f"Knowledge Base initialization error: {self.init_error}")
            return False

    def set_depth(self, k: int):
        self.n_results = k
        return f"RAG search depth set to {k}"

    def clear_index(self):
        if not self._ensure_ready():
            return False, f"Knowledge Base unavailable: {self.init_error}"

        try:
            if self.using_fallback:
                self.vector_store = None
                self._ensure_ready()
                return True, "Knowledge Base cleared."

            self.vector_store.delete_collection()
            # Re-init
            self.vector_store = self._create_vector_store(PERSIST_DIRECTORY)
            return True, "Knowledge Base cleared."
        except Exception as e:
            return False, f"Error clearing KB: {str(e)}"

    def ingest_manual(self, pdf_path: str):
        """
        Loads a PDF manual, splits it into chunks, and stores it in the vector DB.
        """
        if not os.path.exists(pdf_path):
            return False, "File not found."

        if not self._ensure_ready():
            return False, f"Knowledge Base unavailable: {self.init_error}"

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
            
            return True, f"Successfully assimilated {len(chunks)} chunks from manual."
        except Exception as e:
            return False, f"Error ingesting manual: {str(e)}"

            
            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
            
            return True, f"Successfully assimilated {len(chunks)} chunks from manual."
        except Exception as e:
            return False, f"Error ingesting manual: {str(e)}"

    def search_manuals(self, query: str, k=None):
        """
        Retrieves top-k relevant chunks for a given query.
        """
        if k is None: k = self.n_results
        if not self._ensure_ready():
            logger.warning(f"RAG Search Skipped: Knowledge Base unavailable: {self.init_error}")
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"RAG Search Error: {e}", exc_info=True)
            return []

# Singleton instance
kb = KnowledgeBase()
