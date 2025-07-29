"""
Enhanced RAG Pipeline for Education Policy Analysis
==================================================

This module implements an advanced Retrieval-Augmented Generation (RAG) pipeline
specifically designed for education policy research and analysis. It provides
multiple retrieval strategies, comprehensive evaluation capabilities, and
seamless integration with the evaluation system.

Key Features:
- Multiple retrieval strategies (vector, MMR, multi-query, BM25, ensemble, compressed)
- Integrated evaluation and benchmarking capabilities
- Advanced document processing with semantic chunking
- Strategy comparison and optimization tools
- Enterprise-ready error handling and logging

Classes:
    EnhancedRAGPipeline: Main RAG pipeline with advanced retrieval and evaluation

Functions:
    create_enhanced_pipeline(): Factory function for pipeline creation
    ask_question(): Backward-compatible question answering interface

Example Usage:
    >>> from rag_pipeline import EnhancedRAGPipeline
    >>> pipeline = EnhancedRAGPipeline()
    >>> result = pipeline.ask_question(
    ...     "What are the main challenges in education policy?",
    ...     strategy="ensemble"
    ... )
    >>> print(result['answer'])
    
    # Evaluate response quality
    >>> eval_result = pipeline.evaluate_response(
    ...     "How effective are school choice programs?",
    ...     strategy="ensemble"
    ... )
    >>> print(f"Score: {eval_result['overall_score']:.2f}/10")
    
    # Compare strategies
    >>> comparison = pipeline.compare_strategies(
    ...     "What role do teachers play in reform?"
    ... )
    >>> print(f"Best strategy: {comparison['summary']}")

Dependencies:
    - langchain_openai: For ChatOpenAI and language model integration
    - langchain.prompts: For prompt template management
    - langchain.chains: For document processing chains
    - retriever: Custom advanced retriever implementation
    - config: Centralized configuration management
    - evaluation: Comprehensive evaluation system (optional)
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from retriever import AdvancedRetriever
from config import get_config
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with advanced retrieval capabilities and evaluation integration.
    
    This class implements a sophisticated RAG system that supports multiple retrieval
    strategies, comprehensive evaluation metrics, and strategy comparison. It's designed
    for education policy analysis but can be adapted for other domains.
    
    The pipeline supports six different retrieval strategies:
    - Vector: Semantic similarity search using embeddings
    - MMR: Maximum Marginal Relevance for diverse results
    - Multi-Query: Query expansion and refinement
    - BM25: Keyword-based traditional search
    - Ensemble: Hybrid approach combining vector and BM25
    - Compressed: LLM-reranked and filtered results
    
    Attributes:
        config (RAGConfig): System configuration settings
        llm (ChatOpenAI): Language model for answer generation
        retriever (AdvancedRetriever): Multi-strategy document retriever
        rag_prompt (PromptTemplate): Template for RAG prompts
        document_chain: LangChain document processing chain
    
    Methods:
        ask_question(): Ask a question using specified retrieval strategy
        compare_strategies(): Compare performance across different strategies
        evaluate_response(): Evaluate response quality with detailed metrics
        benchmark_pipeline(): Run comprehensive performance benchmarks
        get_pipeline_stats(): Get system statistics and configuration
    
    Example:
        >>> pipeline = EnhancedRAGPipeline()
        >>> result = pipeline.ask_question(
        ...     "What are challenges in education policy implementation?",
        ...     strategy="ensemble",
        ...     max_docs=5,
        ...     include_metadata=True
        ... )
        >>> print(result['answer'])
        >>> print(f"Sources: {result['num_documents']} documents")
        
        # Compare different strategies
        >>> comparison = pipeline.compare_strategies(
        ...     "How effective are charter schools?",
        ...     strategies=["vector", "ensemble", "bm25"]
        ... )
        >>> print(f"Best strategy: {comparison['best_strategy']}")
    
    Note:
        The pipeline requires a properly configured environment with OpenAI API
        access and pre-built vector stores. Use validate_config.py to check setup.
    """
    
    def __init__(self):
        """Initialize the enhanced RAG pipeline."""
        self.config = get_config()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Initialize advanced retriever
        logger.info("Initializing advanced retriever...")
        self.retriever = AdvancedRetriever()
        
        # Create prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "input"],
            template="""You are an expert assistant for education policy question-answering tasks. 
Use the following pieces of retrieved context to answer the question accurately and comprehensively.

Guidelines:
- Provide evidence-based answers using the provided context
- If you don't know the answer, clearly state that you don't know
- Keep answers concise but informative (aim for 2-4 sentences)
- Cite specific information from the context when possible
- When multiple sources provide conflicting information, acknowledge the differences

Context:
{context}

Question:
{input}

Answer:"""
        )
        
        # Create document processing chain
        self.document_chain = create_stuff_documents_chain(self.llm, self.rag_prompt)
        
        logger.info("Enhanced RAG pipeline initialized successfully")
    
    def ask_question(self, 
                    question: str, 
                    strategy: str = "ensemble", 
                    max_docs: Optional[int] = None,
                    include_metadata: bool = True) -> Dict[str, Any]:
        """
        Ask a question using the enhanced RAG pipeline.
        
        Args:
            question: The question to ask
            strategy: Retrieval strategy to use
            max_docs: Maximum number of documents to retrieve
            include_metadata: Whether to include detailed metadata in response
        
        Returns:
            Dictionary containing answer, context, and metadata
        """
        try:
            logger.info(f"Processing question with {strategy} strategy: {question}")
            
            # Retrieve relevant documents
            documents = self.retriever.retrieve_documents(
                query=question, 
                strategy=strategy, 
                max_docs=max_docs
            )
            
            if not documents:
                logger.warning("No documents retrieved for the question")
                return {
                    "answer": "I apologize, but I couldn't find relevant information to answer your question.",
                    "context": [],
                    "strategy_used": strategy,
                    "num_documents": 0
                }
            
            # Generate answer using the document chain
            response = self.document_chain.invoke({
                "input": question,
                "context": documents
            })
            
            # Prepare result
            result = {
                "answer": response,
                "context": documents,
                "strategy_used": strategy,
                "num_documents": len(documents)
            }
            
            # Add detailed metadata if requested
            if include_metadata:
                result.update({
                    "document_sources": [doc.metadata.get('url', 'Unknown') for doc in documents],
                    "document_titles": [doc.metadata.get('title', 'Unknown') for doc in documents],
                    "document_authors": [doc.metadata.get('authors', 'Unknown') for doc in documents],
                    "retrieval_stats": self.retriever.get_retrieval_stats()
                })
            
            logger.info(f"Question processed successfully using {strategy} strategy")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "context": [],
                "error": str(e),
                "strategy_used": strategy,
                "num_documents": 0
            }
    
    def compare_strategies(self, question: str, strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare different retrieval strategies for the same question.
        
        Args:
            question: The question to analyze
            strategies: List of strategies to compare
        
        Returns:
            Dictionary comparing results from different strategies
        """
        if strategies is None:
            strategies = ["vector", "mmr", "ensemble", "compressed"]
        
        logger.info(f"Comparing strategies {strategies} for question: {question}")
        
        comparison_results = {}
        
        for strategy in strategies:
            try:
                result = self.ask_question(question, strategy=strategy, include_metadata=False)
                comparison_results[strategy] = {
                    "answer": result["answer"],
                    "num_documents": result["num_documents"],
                    "document_titles": [doc.metadata.get('title', 'Unknown')[:50] + "..." 
                                      for doc in result["context"]]
                }
            except Exception as e:
                logger.error(f"Error with strategy {strategy}: {e}")
                comparison_results[strategy] = {
                    "answer": f"Error: {str(e)}",
                    "num_documents": 0,
                    "document_titles": []
                }
        
        return {
            "question": question,
            "strategy_comparison": comparison_results,
            "summary": self._generate_strategy_summary(comparison_results)
        }
    
    def _generate_strategy_summary(self, comparison_results: Dict) -> str:
        """Generate a summary of strategy comparison results."""
        successful_strategies = [k for k, v in comparison_results.items() 
                               if not v["answer"].startswith("Error")]
        
        if not successful_strategies:
            return "All strategies failed to process the question."
        
        doc_counts = {k: v["num_documents"] for k, v in comparison_results.items() 
                     if k in successful_strategies}
        
        best_strategy = max(doc_counts.keys(), key=lambda x: doc_counts[x])
        
        return f"Most successful strategy: {best_strategy} ({doc_counts[best_strategy]} documents). " + \
               f"Strategies tested: {', '.join(successful_strategies)}."
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the pipeline."""
        retrieval_stats = self.retriever.get_retrieval_stats()
        
        pipeline_stats = {
            "llm_model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "retrieval_system": retrieval_stats,
            "available_strategies": retrieval_stats["available_strategies"]
        }
        
        return pipeline_stats
    
    def evaluate_response(self, question: str, strategy: str = "ensemble") -> Dict[str, Any]:
        """
        Evaluate a response using the integrated evaluation system.
        
        Args:
            question: The question to evaluate
            strategy: Retrieval strategy to use
        
        Returns:
            Dictionary containing response and evaluation metrics
        """
        try:
            # Import here to avoid circular imports
            from evaluation import RAGEvaluator
            
            # Create evaluator instance
            evaluator = RAGEvaluator(pipeline=self)
            
            # Get evaluation metrics
            metrics = evaluator.evaluate_single_query(question, strategy)
            
            # Also get the actual response
            response = self.ask_question(question, strategy=strategy, include_metadata=True)
            
            return {
                "question": question,
                "strategy": strategy,
                "response": response,
                "evaluation_metrics": metrics.to_dict(),
                "overall_score": metrics.overall_score,
                "performance_summary": {
                    "answer_quality": (metrics.relevance_score + metrics.completeness_score + 
                                     metrics.accuracy_score + metrics.clarity_score) / 4,
                    "retrieval_performance": (metrics.retrieval_precision + metrics.retrieval_recall + 
                                           metrics.document_diversity + metrics.source_coverage) / 4 * 10,
                    "response_time": metrics.response_time
                }
            }
            
        except ImportError:
            logger.warning("Evaluation module not available. Returning response without evaluation.")
            response = self.ask_question(question, strategy=strategy, include_metadata=True)
            return {
                "question": question,
                "strategy": strategy,
                "response": response,
                "evaluation_metrics": None,
                "message": "Evaluation module not available"
            }
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            response = self.ask_question(question, strategy=strategy, include_metadata=True)
            return {
                "question": question,
                "strategy": strategy,
                "response": response,
                "evaluation_metrics": None,
                "error": str(e)
            }
    
    def benchmark_pipeline(self, test_questions: List[str] = None) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark of the pipeline.
        
        Args:
            test_questions: List of questions to test (uses defaults if None)
        
        Returns:
            Dictionary containing benchmark results
        """
        try:
            from evaluation import RAGEvaluator
            
            evaluator = RAGEvaluator(pipeline=self)
            return evaluator.run_benchmark_suite(test_questions)
            
        except ImportError:
            logger.error("Evaluation module not available for benchmarking")
            return {"error": "Evaluation module not available"}
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            return {"error": str(e)}


# Factory function for backward compatibility
def create_enhanced_pipeline() -> EnhancedRAGPipeline:
    """Create and return an EnhancedRAGPipeline instance."""
    return EnhancedRAGPipeline()


# Backward compatibility function
def ask_question(question: str, strategy: str = "ensemble") -> Dict[str, Any]:
    """
    Backward compatible function for the original ask_question interface.
    
    Args:
        question: The question to ask
        strategy: Retrieval strategy to use
    
    Returns:
        Dictionary containing answer and context
    """
    pipeline = EnhancedRAGPipeline()
    return pipeline.ask_question(question, strategy=strategy)


if __name__ == "__main__":
    # Test the enhanced pipeline
    pipeline = EnhancedRAGPipeline()
    
    test_questions = [
        "What are the main challenges in education policy implementation?",
        "How effective are school choice programs?",
        "What role do teachers play in education reform?"
    ]
    
    print("Testing Enhanced RAG Pipeline")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test Question {i} ---")
        print(f"Q: {question}")
        
        # Test with ensemble strategy
        result = pipeline.ask_question(question, strategy="ensemble")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['num_documents']} documents")
        
        # Show source titles
        if result['context']:
            print("Source titles:")
            for j, doc in enumerate(result['context'][:3], 1):
                title = doc.metadata.get('title', 'Unknown')[:60]
                print(f"  {j}. {title}...")
    
    # Test strategy comparison
    print(f"\n--- Strategy Comparison ---")
    comparison = pipeline.compare_strategies(test_questions[0])
    print(f"Question: {comparison['question']}")
    print(f"Summary: {comparison['summary']}")
    
    for strategy, result in comparison['strategy_comparison'].items():
        print(f"\n{strategy.upper()}: {result['num_documents']} docs")
        print(f"Answer preview: {result['answer'][:100]}...")
    
    # Print pipeline stats
    print(f"\n--- Pipeline Statistics ---")
    stats = pipeline.get_pipeline_stats()
    for key, value in stats.items():
        if key != "retrieval_system":
            print(f"{key}: {value}")
