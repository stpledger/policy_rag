import sqlite3
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import get_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_articles(db_path=None):
    """Load articles from database using configuration."""
    config = get_config()
    if db_path is None:
        db_path = config.db_path
    
    logger.info(f"Loading articles from {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM articles", conn)
    conn.close()
    logger.info(f"Loaded {len(df)} articles")
    return df


def chunk_articles(df):
    """Split articles into chunks using semantic chunking."""
    config = get_config()
    logger.info(f"Chunking {len(df)} articles with semantic splitter")
    
    splitter = SemanticChunker(OpenAIEmbeddings(model=config.embedding_model))
    chunks = []
    
    for idx, row in df.iterrows():
        logger.debug(f"Processing article {idx + 1}/{len(df)}: {row['title']}")
        text_chunks = splitter.split_text(row["full_text"])
        
        for chunk in text_chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "title": row["title"],
                    "url": row["url"],
                    "date": row["published_date"],
                    "authors": row["authors"]
                }
            )
            chunks.append(doc)
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks


def build_vectorstore(chunks):
    """Build and save FAISS vectorstore from document chunks."""
    config = get_config()
    logger.info(f"Building vectorstore with {len(chunks)} chunks")
    
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(config.vectorstore_path)
    
    logger.info(f"Vectorstore saved to {config.vectorstore_path}")
    return vectorstore


if __name__ == "__main__":
    print("Loading articles from database...")
    df = load_articles()
    print(f"Loaded {len(df)} articles.")
    print("Splitting articles into chunks...")
    chunks = chunk_articles(df)
    print(f"Created {len(chunks)} text chunks.")
    print("Building and saving FAISS vectorstore...")
    build_vectorstore(chunks)
    print("Vectorstore built and saved as 'ed_policy_vec'.")