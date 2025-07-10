
import requests
import sqlite3
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

# Connect to SQLite DB
conn = sqlite3.connect("finance_news.db")
cursor = conn.cursor()

# Create articles table
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    summary TEXT,
    url TEXT UNIQUE,
    published_date TEXT,
    full_text TEXT
)
""")
conn.commit()


def get_articles_from_newsapi():
    """Fetch today's finance news articles using NewsAPI."""    
    url = "https://newsapi.org/v2/top-headlines"
    if not NEWSAPI_KEY:
        print("ERROR: NEWSAPI_KEY environment variable not set.")
        return []
    params = {
        "country": "us",
        "category": "business",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 100
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"NewsAPI request failed: {r.status_code} {r.text}")
        return []
    data = r.json()
    return data.get("articles", [])


# NewsAPI already provides structured data, so no scraping is needed
def article_to_db_dict(article):
    return {
        "title": article.get("title"),
        "summary": article.get("description"),
        "url": article.get("url"),
        "published_date": article.get("publishedAt"),
        "full_text": article.get("content")
    }

def save_to_db(article_data):
    try:
        cursor.execute("""
        INSERT INTO articles (title, summary, url, published_date, full_text)
        VALUES (?, ?, ?, ?, ?)
        """, (
            article_data["title"],
            article_data["summary"],
            article_data["url"],
            article_data["published_date"],
            article_data["full_text"]
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Duplicate: {article_data['url']}")

# Main execution
if __name__ == "__main__":
    print("Fetching finance news articles from NewsAPI...")
    articles = get_articles_from_newsapi()
    print(f"Found {len(articles)} articles.")

    for article in articles:
        data = article_to_db_dict(article)
        print("---")
        print(f"Title: {data.get('title')}")
        print(f"Summary: {data.get('summary')}")
        print(f"Published: {data.get('published_date')}")
        print(f"URL: {data.get('url')}")
        print(f"Full text (first 500 chars):\n{data.get('full_text')[:500] if data.get('full_text') else ''}")
        
        save_to_db(data)
        time.sleep(1)
