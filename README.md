# 🏛️ Education Policy RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for education policy analysis, featuring advanced retrieval strategies, evaluation metrics, and a modern web interface.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/stpledger/policy_rag.git
   cd policy_rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

4. **Validate setup**
   ```bash
   python scripts/validate.py
   ```

5. **Initialize system (optional)**
   ```bash
   python scripts/setup.py
   ```

### Running the Application

#### Web Interface (Recommended)
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to access the interactive interface with:
- **Ask Questions** tab: Query the system with natural language
- **Evaluation** tab: Analyze response quality with detailed metrics
- **Benchmark** tab: Run comprehensive performance evaluations

#### Python API
```python
from src.core.rag_pipeline import EnhancedRAGPipeline

# Initialize pipeline
pipeline = EnhancedRAGPipeline()

# Ask a question
result = pipeline.ask_question(
    "What are the main challenges in education policy implementation?",
    strategy="ensemble"
)

print(result['answer'])
```

#### Evaluation
```python
from src.evaluation.evaluation import RAGEvaluator

# Evaluate performance
evaluator = RAGEvaluator()
metrics = evaluator.evaluate_single_query(
    "How effective are school choice programs?",
    strategy="ensemble"
)

print(f"Overall Score: {metrics.overall_score:.2f}/10")
```

## 🌟 System Overview

### Advanced RAG Pipeline
- **Multiple Retrieval Strategies**: 6 different approaches including vector search, MMR, multi-query, BM25, ensemble, and compressed retrieval
- **Semantic Chunking**: Intelligent document segmentation using OpenAI embeddings for optimal context preservation
- **Hybrid Search**: Combines semantic and keyword-based retrieval for comprehensive results
- **Strategy Comparison**: Real-time comparison of different retrieval approaches

### Comprehensive Evaluation System
- **Answer Quality Metrics**: Automated scoring for relevance, completeness, accuracy, and clarity (1-10 scale)
- **Retrieval Performance**: Precision, recall, diversity, and source coverage analysis (0-1 scale)
- **Automated Benchmarking**: Multi-question performance evaluation with detailed reports
- **Strategy Optimization**: Data-driven insights for choosing optimal retrieval methods

### Modern Web Interface
- **Interactive Streamlit App**: User-friendly interface with real-time results and visualizations
- **Visual Analytics**: Plotly charts for performance analysis, source exploration, and strategy comparison
- **Multi-Tab Design**: Separate interfaces for questioning, evaluation, and benchmarking
- **Rich Metadata Display**: Detailed source information with author analysis and publication timelines

### Enterprise-Ready Architecture
- **Centralized Configuration**: Environment-based settings with comprehensive validation
- **Error Handling**: Robust error management and detailed logging throughout the system
- **Modular Design**: Clean separation of concerns for easy maintenance and extension
- **Performance Monitoring**: Built-in metrics, timing analysis, and rate limiting awareness

## 🎯 Retrieval Strategies

| Strategy | Description | Best For | Performance |
|----------|-------------|----------|-------------|
| **Ensemble** | Combines vector + BM25 search | General use (recommended) | Highest overall scores |
| **Vector** | Semantic similarity search | Conceptual questions | Great for broad topics |
| **MMR** | Maximum Marginal Relevance | Diverse perspectives | Best for balanced views |
| **Multi-Query** | Query expansion and refinement | Complex, multi-faceted topics | Comprehensive coverage |
| **BM25** | Keyword-based search | Specific terms/names | Fast, precise matching |
| **Compressed** | LLM-reranked results | High-precision needs | Quality over quantity |

## 📊 Data Sources

The system includes **45 education policy articles** from the Brookings Institution, covering:
- Education policy implementation challenges
- School choice and charter school effectiveness  
- Teacher effectiveness and education reform
- Education funding and equity issues
- State and federal education standards
- Technology integration in education

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional (with defaults)
RAG_MODEL_NAME=gpt-4
RAG_TEMPERATURE=0.0
RAG_MAX_TOKENS=2000
RAG_RETRIEVAL_K=5
RAG_EMBEDDING_MODEL=text-embedding-ada-002
```

### Performance Tuning
- **Default Strategy**: `ensemble` (recommended for best overall performance)
- **Document Count**: Adjust `max_docs` parameter (3-10) based on question complexity
- **Caching**: Enabled by default for improved response times
- **Rate Limiting**: Built-in OpenAI API rate limit handling with automatic retries

## 🧪 Testing & Validation

### Run Evaluation Tests
```bash
python test_evaluation.py
```

### Validate Configuration
```bash
python validate_config.py
```

### Interactive Development
```bash
jupyter notebook scratchpad.ipynb
```

## 📁 Project Structure

```
policy_rag/
├── 📄 app.py                    # Streamlit web application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
│
├── 📦 src/                      # Source code modules
│   ├── core/                    # Core RAG components
│   │   ├── config.py            # Configuration management
│   │   ├── rag_pipeline.py      # Enhanced RAG pipeline
│   │   └── retriever.py         # Advanced retrieval strategies
│   ├── data/                    # Data processing modules
│   │   ├── scrape.py            # Web scraping system
│   │   └── vectorize.py         # Document vectorization
│   ├── evaluation/              # Evaluation system
│   │   └── evaluation.py        # Comprehensive evaluation
│   └── utils/                   # Utility modules
│       └── validate_config.py   # Configuration validation
│
├── 🧪 tests/                    # Test suite
│   └── test_evaluation.py       # Evaluation system tests
│
├── 📊 data/                     # Data storage
│   ├── raw/                     # Raw scraped data
│   │   ├── main.db              # SQLite database
│   │   └── links.txt            # Source URLs
│   ├── processed/               # Processed data
│   │   └── ed_policy_vec/       # FAISS vectorstore
│   └── docs/                    # Source documents
│       └── *.pdf                # Brookings policy papers
│
├── 🔧 scripts/                  # Utility scripts
│   ├── setup.py                 # System initialization
│   ├── validate.py              # System validation
│   └── benchmark.py             # Performance testing
│
├── 📝 notebooks/                # Development notebooks
│   └── scratchpad.ipynb         # Development notebook
│
└── 🗃️ archive/                  # Backup files
    └── rag_pipeline.py          # Original implementation
```

## 📈 Performance Benchmarks

### Example Results
- **Average Overall Score**: 7.8/10
- **Answer Quality**: 8.1/10 (relevance, completeness, accuracy, clarity)
- **Retrieval Precision**: 0.73 (proportion of relevant documents)
- **Average Response Time**: 2.4 seconds

### Optimization Tips
1. Use `ensemble` strategy for best overall performance
2. Adjust `max_docs` parameter based on question complexity
3. Enable caching for repeated queries
4. Monitor API usage to avoid rate limiting

## 🛠️ Development & Extension

### Adding New Retrieval Strategies
1. Implement strategy in `retriever.py`
2. Add to strategy list in `AdvancedRetriever`
3. Update documentation and tests

### Custom Evaluation Metrics  
1. Extend `EvaluationMetrics` dataclass
2. Add evaluation logic in `RAGEvaluator`
3. Update UI components in `app.py`

### Extending Data Sources
1. Add URLs to `links.txt`
2. Run `python scrape.py`
3. Run `python vectorize.py`
4. Restart application

## 🧱 Technology Stack

- **Python 3.10+** with modern async/await patterns
- **OpenAI API** for embeddings (text-embedding-ada-002) and LLM (GPT-4)
- **LangChain** for RAG orchestration and document processing
- **FAISS** for efficient vector similarity search
- **Streamlit** for interactive web interface
- **Plotly** for rich data visualizations
- **SQLite** for document storage and metadata
- **BeautifulSoup** for web scraping and content extraction