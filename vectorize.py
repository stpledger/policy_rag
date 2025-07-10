import sqlite3
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_articles(db_path="finance_news.db", min_length=200):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM articles", conn)
    # Filter out very short entries
    df = df[df["full_text"].str.len() > min_length]
    return df


def chunk_articles(df):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []

    for _, row in df.iterrows():
        text_chunks = splitter.split_text(row["full_text"])
        for chunk in text_chunks:
            chunks.append({"content": chunk, "metadata": {"title": row["title"], "url": row["url"]}})
    
    return chunks


def build_vectorstore(chunks):
    texts = [chunk["content"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]
    
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
    vectorstore.save_local("faiss_finance_news")

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
    print("Vectorstore built and saved as 'faiss_finance_news'.")