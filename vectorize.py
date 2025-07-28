import sqlite3
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def load_articles(db_path="main.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM articles", conn)
    return df


def chunk_articles(df):
    splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = []
    for _, row in df.iterrows():
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
    return chunks


def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local("ed_policy_vec")
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