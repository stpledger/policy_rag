import requests
from bs4 import BeautifulSoup
import re
import sqlite3

# Connect to SQLite DB
conn = sqlite3.connect("main.db")
cursor = conn.cursor()

# Create articles table
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authors TEXT,
    key_points TEXT,
    url TEXT UNIQUE,
    published_date TEXT,
    full_text TEXT
)
""")
conn.commit()


def clean_html_text(text):
    """Remove excessive whitespace from text."""
    return re.sub(r'\s+', ' ', text).strip()

def parse_brookings_article(url):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Title
    title = ''
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)
        if title.endswith(' | Brookings'):
            title = title[:-len(' | Brookings')]

    # Authors
    authors = []
    seen = set()
    for name in soup.select('div.people-wrapper span.name'):
        author = name.get_text(strip=True)
        if author and author not in seen:
            authors.append(author)
            seen.add(author)

    # Date
    date = ''
    date_pattern = re.compile(r'([A-Z][a-z]+ \d{1,2}, \d{4})')
    search_areas = [soup.find('article'), soup.find('div', class_='people-wrapper')]
    for area in search_areas:
        if area:
            match = date_pattern.search(area.get_text(" ", strip=True))
            if match:
                date = match.group(1)
                break
    if not date:
        match = date_pattern.search(soup.get_text(" ", strip=True))
        if match:
            date = match.group(1)

    # Key Points
    key_points = []
    ul_tag = soup.find('div', class_='people-wrapper')
    if ul_tag:
        ul_tag = ul_tag.find_next('ul')
        if ul_tag:
            key_points = [li.get_text(strip=True) for li in ul_tag.find_all('li')]

    # Article Body: <p> tags after key points <ul>, skip 'group-list' class
    paragraphs = []
    if ul_tag:
        for sib in ul_tag.find_all_next('p'):
            if not sib.find_parent(['script', 'style']) and 'group-list' not in (sib.get('class') or []):
                paragraphs.append(sib)
    else:
        paragraphs = [p for p in soup.find_all('p') if not p.find_parent(['script', 'style']) and 'group-list' not in (p.get('class') or [])]
    raw_body = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    related_idx = raw_body.lower().find('related content')
    if related_idx != -1:
        raw_body = raw_body[:related_idx].rstrip()
    article_body = clean_html_text(raw_body)

    return {
        'title': title,
        'date': date,
        'authors': authors,
        'key_points': key_points,
        'body': article_body,
        'url': url,
    }

def save_to_db(article_data):
    try:
        cursor.execute("""
        INSERT INTO articles (title, authors, key_points, url, published_date, full_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            article_data["title"],
            ", ".join(article_data["authors"]),
            "\n".join(article_data["key_points"]),
            article_data["url"],
            article_data["date"],
            article_data["body"]
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Duplicate: {article_data['url']}")

if __name__ == "__main__":
    with open('links.txt', 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    for url in urls:
        print(f"Processing: {url}")
        try:
            data = parse_brookings_article(url)
            print(f"Title: {data['title']}")
            print(f"Date: {data['date']}")
            print(f"Authors: {', '.join(data['authors'])}")
            print("--- Key Points ---")
            for pt in data['key_points']:
                print(f"- {pt}")
            print("\n--- End of Content Preview ---\n")
            save_to_db(data)
        except Exception as e:
            print(f"Failed to process {url}: {e}")
