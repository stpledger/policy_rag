# 🧠📈 Finance News RAG Model (Proof of Concept)

This project is a lightweight prototype of a Retrieval-Augmented Generation (RAG) system that can read recent financial and business news and intelligently answer natural language questions about it.

⚠️ **Note**: NewsAPI only provides content previews (not full article text), which may limit the detail available for question answering.

🔮 **Future extension idea**: Automatically deliver personalized summaries of financial news based on an individual’s or organization’s investment portfolio.

🔗[Demo Website](https://finance-rag.streamlit.app/)

---

## ⚙️ How It Works

### 📰 `scrape.py`
1. Finance articles are retrieved from [NewsAPI](https://newsapi.org/).
2. Article metadata and content are saved into a local SQLite database.

### 🧩 `vectorize.py`
3. Articles are broken into overlapping chunks (500 characters each, with 50 characters of overlap) to preserve context during search.
4. Each chunk is embedded using OpenAI's `text-embedding-ada-002` model via the OpenAI API.
5. The resulting vectors are stored in a FAISS (Facebook AI Similarity Search) index for fast semantic retrieval.

### 🤖 `rag_pipeline.py`
6. The user enters a natural language question.
7. A retriever searches the vectorstore to find the most semantically relevant article chunks.
8. These chunks, along with the question, are inserted into a custom prompt template that guides an LLM to generate an answer.
9. The LLM responds with a grounded, context-aware answer based on the retrieved content.

---

## 🔍 Example

The below example can be seen in the [scratchpad.ipynb](scratchpad.ipynb) file.

> _"What are the latest trends in the stock market?"_

Relevant Context:

    Title: S&P 500 rebounds after 2 days of losses, Nvidia leads gain as it reaches $4 trillion market value: Live updates - CNBC
    URL: https://www.cnbc.com/2025/07/08/stock-market-today-live-updates.html
    Content: The S&P 500 rose Wednesday, led by tech, as Nvidia reached a major milestone and investors monitored the latest tariff updates from President Donald Trump. The broad market benchmark climbed 0.3…

    Title: Asia Set for Cautious Open as Trump Doubles Down: Markets Wrap - Bloomberg
    URL: https://www.bloomberg.com/news/articles/2025-07-08/stock-market-today-dow-s-p-live-updates
    Content: A rally in several big techs spurred a rebound in stocks, with Nvidia Corp. briefly hitting $4 trillion - the first company in history to achieve that milestone. Treasuries rose before a $39 billion …

    Title: Bank of America delivers bold S&P 500 target - TheStreet
    URL: https://www.thestreet.com/investing/bank-of-america-delivers-bold-s-p-500-target
    Content: The S&P 500 is widely considered the benchmark index most investors use to measure performance for good reason. It includes 500 of the largest companies in America, crisscrossing sectors and indu…

    Title: Treasury yields move lower as investors monitor latest tariff news - CNBC
    URL: https://www.cnbc.com/2025/07/09/treasury-yields-move-lower-as-investors-monitor-latest-tariff-news.html
    Content: U.S. Treasury yields were lower on Wednesday as investors monitored the latest tariff developments after President Donald Trump sent letters dictating new tariff rates to at least 14 countries. The …

> Answer: _The latest trends in the stock market include a rise in the S&P 500, led by tech companies such as Nvidia Corp. which recently hit a major milestone by briefly reaching a $4 trillion valuation. Additionally, U.S. Treasury yields were lower as investors monitored the latest tariff developments._

---

## 🧱 Stack

- `Python 3.10`
- `OpenAI API` for embeddings + GPT-4
- `FAISS` for vector search
- `LangChain` for orchestration
- `SQLite` for document storage
- `NewsAPI` for article ingestion
- `Streamlit Community Cloud` for deployment