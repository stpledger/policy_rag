import streamlit as st
from rag_pipeline import EnhancedRAGPipeline
from config import get_config
import pandas as pd
import plotly.express as px

# Get configuration
config = get_config()

# Initialize the enhanced pipeline
@st.cache_resource
def load_pipeline():
    """Load and cache the RAG pipeline."""
    return EnhancedRAGPipeline()

pipeline = load_pipeline()

# Set page config
st.set_page_config(
    page_title="Education Policy RAG System",
    page_icon="🏛️",
    layout="wide"
)

st.title(config.app_title)
st.markdown("*Advanced RAG system for education policy analysis with multiple retrieval strategies*")

# Sidebar for settings and information
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Strategy selection
    strategy = st.selectbox(
        "Select Retrieval Strategy:",
        options=["ensemble", "vector", "mmr", "multi_query", "bm25", "compressed"],
        index=0,
        help="""
        - **ensemble**: Combines vector and keyword search (recommended)
        - **vector**: Semantic similarity search
        - **mmr**: Maximum Marginal Relevance for diversity
        - **multi_query**: Expands query for better coverage
        - **bm25**: Keyword-based search
        - **compressed**: LLM-reranked results
        """
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        max_docs = st.slider("Max Documents", 3, 10, 5)
        include_metadata = st.checkbox("Include Metadata", True)
        show_strategy_comparison = st.checkbox("Compare Strategies", False)
    
    # System information
    st.header("📊 System Info")
    stats = pipeline.get_pipeline_stats()
    st.metric("LLM Model", stats["llm_model"])
    st.metric("Available Strategies", len(stats["available_strategies"]))
    st.metric("Vectorstore Size", stats["retrieval_system"]["vectorstore_size"])

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask a Question")
    
    # Example questions
    example_questions = [
        "What are the main challenges in education policy implementation?",
        "How effective are school choice programs?",
        "What role do teachers play in education reform?",
        "What are the latest trends in education funding?",
        "How do different states approach education standards?"
    ]
    
    # Question input with examples
    if 'selected_question' in st.session_state:
        question = st.text_input(
            "Ask a question about education policy:",
            value=st.session_state.selected_question,
            placeholder="e.g., What are the challenges in implementing education policy?"
        )
        del st.session_state.selected_question
    else:
        question = st.text_input(
            "Ask a question about education policy:",
            placeholder="e.g., What are the challenges in implementing education policy?"
        )
    
    # Example question buttons
    st.write("**Example questions:**")
    cols = st.columns(2)
    for i, eq in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {eq[:40]}...", key=f"example_{i}"):
                st.session_state.selected_question = eq

with col2:
    st.header("🎯 Quick Actions")
    
    if st.button("🔄 Refresh Pipeline", help="Reload the RAG pipeline"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("📈 Show System Stats"):
        st.json(stats)

# Process question
if question:
    if show_strategy_comparison:
        st.header("🔍 Strategy Comparison")
        
        with st.spinner("Comparing different retrieval strategies..."):
            comparison = pipeline.compare_strategies(question)
        
        st.subheader(f"Question: {comparison['question']}")
        st.info(comparison['summary'])
        
        # Display results for each strategy
        for strategy_name, result in comparison['strategy_comparison'].items():
            with st.expander(f"📊 {strategy_name.upper()} Strategy ({result['num_documents']} docs)"):
                st.write("**Answer:**")
                st.write(result['answer'])
                
                if result['document_titles']:
                    st.write("**Sources:**")
                    for title in result['document_titles']:
                        st.write(f"• {title}")
    
    else:
        st.header("💡 Answer")
        
        with st.spinner(f"Searching with {strategy} strategy..."):
            result = pipeline.ask_question(
                question, 
                strategy=strategy, 
                max_docs=max_docs,
                include_metadata=include_metadata
            )
        
        # Display answer
        st.write(result.get('answer'))
        
        # Display strategy used and document count
        st.caption(f"Used {result.get('strategy_used', 'unknown')} strategy • {result.get('num_documents', 0)} documents retrieved")
        
        # Display sources
        sources = result.get('context', [])
        if sources:
            st.subheader("📚 Sources")
            
            # Create tabs for different source views
            tab1, tab2 = st.tabs(["📋 Source List", "📊 Source Analysis"])
            
            with tab1:
                seen_urls = set()
                unique_sources = []
                for doc in sources:
                    url = doc.metadata.get('url')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_sources.append(doc)
                
                for i, doc in enumerate(unique_sources, 1):
                    with st.expander(f"📄 Source {i}: {doc.metadata.get('title', 'Unknown')[:50]}..."):
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            st.write(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                            st.write(f"**Authors:** {doc.metadata.get('authors', 'N/A')}")
                            st.write(f"**Date:** {doc.metadata.get('date', 'N/A')}")
                        
                        with col_b:
                            if doc.metadata.get('url'):
                                st.link_button("🔗 View Source", doc.metadata['url'])
                        
                        st.write("**Excerpt:**")
                        excerpt = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        st.write(excerpt)
            
            with tab2:
                if include_metadata and 'document_authors' in result:
                    # Author analysis
                    authors = [author for author in result['document_authors'] if author != 'Unknown']
                    if authors:
                        author_counts = pd.Series(authors).value_counts()
                        fig = px.bar(
                            x=author_counts.values, 
                            y=author_counts.index, 
                            orientation='h',
                            title="Sources by Author"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Publication dates
                    dates = [date for date in result.get('document_sources', []) if date != 'Unknown']
                    if dates:
                        st.write("**Publication Timeline:**")
                        date_df = pd.DataFrame({'dates': dates})
                        st.write(date_df['dates'].value_counts().sort_index())

# Footer
st.markdown("---")
st.markdown("*Built with LangChain, OpenAI, and Streamlit*")