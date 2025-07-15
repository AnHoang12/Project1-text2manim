import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OUTPUT_DIR = "output"
    THEOREMS_PATH = os.path.join("data", "easy_20.json")
    CONTEXT_LEARNING_PATH = "data/context_learning"
    CHROMA_DB_PATH = "data/rag/chroma_db"
    MANIM_DOCS_PATH = "data/rag/manim_docs"
    EMBEDDING_MODEL = "azure/text-embedding-3-large"
   