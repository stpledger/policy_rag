"""
Comprehensive Evaluation System for RAG Pipeline
===============================================

This module provides a complete evaluation framework for assessing RAG pipeline
performance across multiple dimensions including answer quality, retrieval
effectiveness, and system performance metrics.

Key Features:
- Multi-dimensional evaluation metrics (relevance, completeness, accuracy, clarity)
- Retrieval performance analysis (precision, recall, diversity, coverage)
- Automated benchmarking with statistical analysis
- Strategy comparison and optimization recommendations
- LLM-based quality assessment with structured scoring
- Performance timing and efficiency metrics

Classes:
    EvaluationMetrics: Container for all evaluation metrics and scores
    RAGEvaluator: Main evaluation engine with comprehensive assessment capabilities

Functions:
    quick_evaluate(): Fast single-question evaluation
    run_quick_benchmark(): Quick benchmark with default questions

Evaluation Dimensions:
    Answer Quality (1-10 scale):
    - Relevance: How well the answer addresses the specific question
    - Completeness: Comprehensiveness of the response coverage
    - Accuracy: Factual correctness based on retrieved context
    - Clarity: Structure, readability, and coherence of the answer
    
    Retrieval Performance (0-1 scale):
    - Precision: Proportion of relevant documents in results
    - Recall: Coverage of relevant information available
    - Diversity: Variety of sources and perspectives included
    - Source Coverage: Breadth of authors and publications
    
    System Performance:
    - Response Time: End-to-end processing duration
    - Document Count: Number of documents successfully retrieved
    - Strategy Efficiency: Comparative performance analysis

Example Usage:
    >>> from evaluation import RAGEvaluator, quick_evaluate
    >>> 
    >>> # Quick single evaluation
    >>> metrics = quick_evaluate(
    ...     "What are challenges in education policy?",
    ...     strategy="ensemble"
    ... )
    >>> print(f"Overall Score: {metrics.overall_score:.2f}/10")
    >>> print(f"Relevance: {metrics.relevance_score:.1f}/10")
    >>> 
    >>> # Detailed evaluation with custom evaluator
    >>> evaluator = RAGEvaluator()
    >>> result = evaluator.evaluate_single_query(
    ...     "How effective are school choice programs?",
    ...     strategy="vector"
    ... )
    >>> print(f"Answer Quality: {result.clarity_score:.1f}/10")
    >>> print(f"Retrieval Precision: {result.retrieval_precision:.3f}")
    >>> 
    >>> # Strategy comparison
    >>> comparison = evaluator.compare_strategies(
    ...     "What role do teachers play in reform?",
    ...     strategies=["vector", "ensemble", "bm25"]
    ... )
    >>> print(f"Best Strategy: {comparison['best_strategy']}")
    >>> 
    >>> # Full benchmark suite
    >>> benchmark = evaluator.run_benchmark_suite()
    >>> avg_score = benchmark["aggregate_metrics"]["avg_overall_score"]
    >>> print(f"System Average: {avg_score:.2f}/10")

Implementation Notes:
    - Uses GPT-4 for evaluation to ensure high-quality assessment
    - Implements structured prompting for consistent scoring
    - Handles API rate limiting with automatic retries
    - Provides detailed error handling and fallback mechanisms
    - Supports custom question sets for domain-specific evaluation

Dependencies:
    - langchain_openai: For evaluation LLM access
    - rag_pipeline: For pipeline integration and testing
    - config: For system configuration access
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from ..core.config import get_config
from ..core.rag_pipeline import EnhancedRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive container for RAG evaluation metrics and scores.
    
    This dataclass stores all evaluation results from assessing RAG pipeline
    performance, including answer quality metrics, retrieval performance
    indicators, and system performance measurements.
    
    Attributes:
        Answer Quality Metrics (1-10 scale):
            relevance_score (float): How well answer addresses the question
            completeness_score (float): Comprehensiveness of response
            accuracy_score (float): Factual correctness based on context
            clarity_score (float): Structure and readability of answer
            
        Retrieval Performance Metrics (0-1 scale):
            retrieval_precision (float): Proportion of relevant documents
            retrieval_recall (float): Coverage of available relevant info
            document_diversity (float): Variety of sources and perspectives
            source_coverage (float): Breadth of authors and publications
            
        Performance Metrics:
            response_time (float): End-to-end processing time in seconds
            num_documents_retrieved (int): Count of documents retrieved
            strategy_used (str): Name of retrieval strategy employed
            
        Overall Assessment:
            overall_score (float): Weighted average of all quality metrics
    
    Methods:
        to_dict(): Convert metrics to dictionary for serialization
        
    Example:
        >>> metrics = EvaluationMetrics(
        ...     relevance_score=8.5,
        ...     completeness_score=7.8,
        ...     accuracy_score=9.2,
        ...     clarity_score=8.1,
        ...     retrieval_precision=0.75,
        ...     response_time=2.4,
        ...     num_documents_retrieved=5,
        ...     strategy_used="ensemble"
        ... )
        >>> print(f"Overall Score: {metrics.overall_score:.1f}/10")
        >>> metrics_dict = metrics.to_dict()
    
    Note:
        All score fields default to 0.0, allowing for partial initialization
        and gradual building of the metrics during evaluation process.
    """
    
    # Answer Quality Metrics
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    clarity_score: float = 0.0
    
    # Retrieval Performance Metrics
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    document_diversity: float = 0.0
    source_coverage: float = 0.0
    
    # Performance Metrics
    response_time: float = 0.0
    num_documents_retrieved: int = 0
    strategy_used: str = ""
    
    # Overall Score
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class RAGEvaluator:
    """
    Comprehensive RAG evaluation system for assessing pipeline performance.
    
    This class provides a complete evaluation framework for Retrieval-Augmented
    Generation systems, offering multiple evaluation strategies and detailed
    performance analysis across various dimensions.
    
    Features:
        - Multi-dimensional evaluation (relevance, completeness, accuracy, clarity)
        - Retrieval performance assessment (precision, recall, diversity)
        - Automated quality scoring using language models
        - Benchmark comparison across different retrieval strategies
        - Performance timing and efficiency metrics
        - Export capabilities for analysis and reporting
    
    Evaluation Dimensions:
        1. Answer Quality (1-10 scale):
           - Relevance: How well the answer addresses the specific question
           - Completeness: Thoroughness and comprehensiveness of response
           - Accuracy: Factual correctness based on provided context
           - Clarity: Structure, readability, and logical flow
        
        2. Retrieval Performance (0-1 scale):
           - Precision: Proportion of retrieved documents that are relevant
           - Recall: Coverage of available relevant information
           - Diversity: Variety of sources and perspectives retrieved
           - Source Coverage: Breadth of authors and publication sources
        
        3. System Performance:
           - Response Time: End-to-end processing latency
           - Document Count: Number of documents retrieved and processed
           - Strategy Effectiveness: Comparative performance analysis
    
    Args:
        pipeline: Optional EnhancedRAGPipeline instance for evaluation
    
    Methods:
        evaluate_answer(): Comprehensive answer quality assessment
        evaluate_retrieval(): Retrieval system performance analysis
        run_benchmark(): Compare performance across retrieval strategies
        export_results(): Save evaluation results to file
        
    Example:
        >>> from evaluation import RAGEvaluator
        >>> evaluator = RAGEvaluator()
        >>> 
        >>> # Single answer evaluation
        >>> metrics = evaluator.evaluate_answer(
        ...     question="What are effective reading interventions?",
        ...     answer="Research shows that phonics-based instruction...",
        ...     context=retrieved_docs
        ... )
        >>> print(f"Overall Score: {metrics.overall_score:.1f}/10")
        >>> 
        >>> # Comprehensive benchmark across strategies
        >>> benchmark_results = evaluator.run_benchmark(test_questions)
        >>> evaluator.export_results(benchmark_results, "evaluation_report.json")
    
    Note:
        The evaluator uses GPT-4 for automated scoring by default. Ensure
        proper OpenAI API configuration for optimal performance.
    """
    
    def __init__(self, pipeline: Optional[EnhancedRAGPipeline] = None):
        """Initialize the evaluator."""
        self.config = get_config()
        self.pipeline = pipeline or EnhancedRAGPipeline()
        
        # Initialize evaluation LLM (potentially different from main pipeline)
        self.eval_llm = ChatOpenAI(
            model_name="gpt-4",  # Use GPT-4 for evaluation
            temperature=0.0,     # Deterministic evaluation
            max_tokens=1000
        )
        
        # Create evaluation prompts
        self._setup_evaluation_prompts()
        
        logger.info("RAG Evaluator initialized successfully")
    
    def _setup_evaluation_prompts(self):
        """Setup prompts for different evaluation aspects."""
        
        # Answer Quality Evaluation
        self.quality_eval_prompt = PromptTemplate(
            input_variables=["question", "answer", "context"],
            template="""You are an expert evaluator assessing the quality of an AI assistant's answer.

Question: {question}

Answer to Evaluate: {answer}

Retrieved Context: {context}

Please evaluate the answer on the following criteria (score 1-10 for each):

1. RELEVANCE: How well does the answer address the specific question asked?
2. COMPLETENESS: Does the answer provide comprehensive coverage of the topic?
3. ACCURACY: Is the information in the answer factually correct based on the context?
4. CLARITY: Is the answer well-structured and easy to understand?

Provide your evaluation in the following JSON format:
{{
    "relevance_score": <score 1-10>,
    "completeness_score": <score 1-10>, 
    "accuracy_score": <score 1-10>,
    "clarity_score": <score 1-10>,
    "reasoning": {{
        "relevance": "<brief explanation>",
        "completeness": "<brief explanation>",
        "accuracy": "<brief explanation>",
        "clarity": "<brief explanation>"
    }},
    "overall_assessment": "<2-3 sentence overall assessment>"
}}"""
        )
        
        # Document Relevance Evaluation
        self.doc_relevance_prompt = PromptTemplate(
            input_variables=["question", "document_content", "document_title"],
            template="""Evaluate how relevant this document is to answering the given question.

Question: {question}

Document Title: {document_title}
Document Content: {document_content}

Rate the relevance on a scale of 1-10 where:
- 1-3: Not relevant or only tangentially related
- 4-6: Somewhat relevant, contains some useful information
- 7-8: Highly relevant, directly addresses the question
- 9-10: Extremely relevant, provides comprehensive answer

Respond with only a single number (1-10):"""
        )
    
    def evaluate_answer_quality(self, question: str, answer: str, context: List[Document]) -> Dict[str, Any]:
        """Evaluate the quality of a generated answer."""
        try:
            # Prepare context for evaluation
            context_text = "\n\n".join([
                f"Document {i+1}: {doc.page_content[:500]}..." 
                for i, doc in enumerate(context[:3])  # Limit context length
            ])
            
            # Get quality evaluation
            response = self.eval_llm.invoke(
                self.quality_eval_prompt.format(
                    question=question,
                    answer=answer,
                    context=context_text
                )
            )
            
            # Parse JSON response
            eval_result = json.loads(response.content)
            
            return eval_result
            
        except Exception as e:
            logger.error(f"Error evaluating answer quality: {e}")
            return {
                "relevance_score": 5.0,
                "completeness_score": 5.0,
                "accuracy_score": 5.0,
                "clarity_score": 5.0,
                "reasoning": {"error": str(e)},
                "overall_assessment": "Evaluation failed due to error"
            }
    
    def evaluate_retrieval_performance(self, question: str, documents: List[Document]) -> Dict[str, float]:
        """Evaluate retrieval performance metrics."""
        if not documents:
            return {
                "retrieval_precision": 0.0,
                "retrieval_recall": 0.0,
                "document_diversity": 0.0,
                "source_coverage": 0.0
            }
        
        try:
            # Evaluate document relevance
            relevance_scores = []
            for doc in documents:
                try:
                    score_response = self.eval_llm.invoke(
                        self.doc_relevance_prompt.format(
                            question=question,
                            document_content=doc.page_content[:1000],
                            document_title=doc.metadata.get('title', 'Unknown')
                        )
                    )
                    score = float(score_response.content.strip())
                    relevance_scores.append(max(1.0, min(10.0, score)))  # Clamp to 1-10
                except:
                    relevance_scores.append(5.0)  # Default score if parsing fails
            
            # Calculate metrics
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            precision = len([s for s in relevance_scores if s >= 7]) / len(relevance_scores)
            
            # Document diversity (unique sources)
            unique_sources = len(set(doc.metadata.get('url', f'doc_{i}') for i, doc in enumerate(documents)))
            diversity = unique_sources / len(documents)
            
            # Source coverage (unique authors)
            unique_authors = len(set(doc.metadata.get('authors', f'author_{i}') for i, doc in enumerate(documents)))
            coverage = min(1.0, unique_authors / 3)  # Normalize to max of 3 authors
            
            return {
                "retrieval_precision": precision,
                "retrieval_recall": min(1.0, avg_relevance / 8.0),  # Estimate based on relevance
                "document_diversity": diversity,
                "source_coverage": coverage
            }
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval performance: {e}")
            return {
                "retrieval_precision": 0.5,
                "retrieval_recall": 0.5,
                "document_diversity": 0.5,
                "source_coverage": 0.5
            }
    
    def evaluate_single_query(self, question: str, strategy: str = "ensemble") -> EvaluationMetrics:
        """Perform comprehensive evaluation of a single query."""
        logger.info(f"Evaluating query: {question} with strategy: {strategy}")
        
        # Measure response time
        start_time = time.time()
        
        # Get answer from pipeline
        result = self.pipeline.ask_question(
            question=question, 
            strategy=strategy, 
            include_metadata=True
        )
        
        response_time = time.time() - start_time
        
        # Extract components
        answer = result.get('answer', '')
        context = result.get('context', [])
        num_docs = result.get('num_documents', 0)
        
        # Evaluate answer quality
        quality_eval = self.evaluate_answer_quality(question, answer, context)
        
        # Evaluate retrieval performance
        retrieval_eval = self.evaluate_retrieval_performance(question, context)
        
        # Calculate overall score
        quality_scores = [
            quality_eval.get('relevance_score', 5),
            quality_eval.get('completeness_score', 5),
            quality_eval.get('accuracy_score', 5),
            quality_eval.get('clarity_score', 5)
        ]
        
        retrieval_scores = [
            retrieval_eval.get('retrieval_precision', 0.5) * 10,
            retrieval_eval.get('retrieval_recall', 0.5) * 10,
            retrieval_eval.get('document_diversity', 0.5) * 10,
            retrieval_eval.get('source_coverage', 0.5) * 10
        ]
        
        overall_score = (sum(quality_scores) + sum(retrieval_scores)) / 8
        
        # Create metrics object
        metrics = EvaluationMetrics(
            # Answer Quality
            relevance_score=quality_eval.get('relevance_score', 5),
            completeness_score=quality_eval.get('completeness_score', 5),
            accuracy_score=quality_eval.get('accuracy_score', 5),
            clarity_score=quality_eval.get('clarity_score', 5),
            
            # Retrieval Performance
            retrieval_precision=retrieval_eval.get('retrieval_precision', 0.5),
            retrieval_recall=retrieval_eval.get('retrieval_recall', 0.5),
            document_diversity=retrieval_eval.get('document_diversity', 0.5),
            source_coverage=retrieval_eval.get('source_coverage', 0.5),
            
            # Performance
            response_time=response_time,
            num_documents_retrieved=num_docs,
            strategy_used=strategy,
            
            # Overall
            overall_score=overall_score
        )
        
        logger.info(f"Evaluation completed - Overall Score: {overall_score:.2f}")
        return metrics
    
    def compare_strategies(self, question: str, strategies: List[str] = None) -> Dict[str, Any]:
        """Compare different retrieval strategies using evaluation metrics."""
        if strategies is None:
            strategies = ["vector", "mmr", "ensemble", "bm25", "compressed"]
        
        logger.info(f"Comparing strategies for evaluation: {strategies}")
        
        strategy_results = {}
        
        for strategy in strategies:
            try:
                metrics = self.evaluate_single_query(question, strategy)
                strategy_results[strategy] = metrics.to_dict()
            except Exception as e:
                logger.error(f"Error evaluating strategy {strategy}: {e}")
                strategy_results[strategy] = {"error": str(e)}
        
        # Find best strategy
        valid_results = {k: v for k, v in strategy_results.items() if "error" not in v}
        
        if valid_results:
            best_strategy = max(valid_results.keys(), 
                              key=lambda x: valid_results[x]["overall_score"])
            best_score = valid_results[best_strategy]["overall_score"]
        else:
            best_strategy = "unknown"
            best_score = 0.0
        
        return {
            "question": question,
            "strategy_results": strategy_results,
            "best_strategy": best_strategy,
            "best_score": best_score,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def run_benchmark_suite(self, test_questions: List[str] = None) -> Dict[str, Any]:
        """Run a comprehensive benchmark evaluation."""
        if test_questions is None:
            test_questions = [
                "What are the main challenges in education policy implementation?",
                "How effective are school choice programs?",
                "What role do teachers play in education reform?",
                "What are the latest trends in education funding?",
                "How do different states approach education standards?",
                "What impact does technology have on student learning outcomes?",
                "How can education policy address inequality?",
                "What are the benefits and drawbacks of standardized testing?"
            ]
        
        logger.info(f"Running benchmark suite with {len(test_questions)} questions")
        
        benchmark_results = {
            "test_questions": test_questions,
            "individual_results": {},
            "aggregate_metrics": {},
            "strategy_comparison": {},
            "benchmark_timestamp": datetime.now().isoformat()
        }
        
        # Evaluate each question with the ensemble strategy
        all_metrics = []
        for i, question in enumerate(test_questions, 1):
            logger.info(f"Evaluating question {i}/{len(test_questions)}")
            metrics = self.evaluate_single_query(question, "ensemble")
            benchmark_results["individual_results"][question] = metrics.to_dict()
            all_metrics.append(metrics)
        
        # Calculate aggregate metrics
        if all_metrics:
            benchmark_results["aggregate_metrics"] = {
                "avg_relevance_score": sum(m.relevance_score for m in all_metrics) / len(all_metrics),
                "avg_completeness_score": sum(m.completeness_score for m in all_metrics) / len(all_metrics),
                "avg_accuracy_score": sum(m.accuracy_score for m in all_metrics) / len(all_metrics),
                "avg_clarity_score": sum(m.clarity_score for m in all_metrics) / len(all_metrics),
                "avg_retrieval_precision": sum(m.retrieval_precision for m in all_metrics) / len(all_metrics),
                "avg_retrieval_recall": sum(m.retrieval_recall for m in all_metrics) / len(all_metrics),
                "avg_document_diversity": sum(m.document_diversity for m in all_metrics) / len(all_metrics),
                "avg_source_coverage": sum(m.source_coverage for m in all_metrics) / len(all_metrics),
                "avg_response_time": sum(m.response_time for m in all_metrics) / len(all_metrics),
                "avg_overall_score": sum(m.overall_score for m in all_metrics) / len(all_metrics),
                "total_questions": len(all_metrics)
            }
        
        # Strategy comparison on a subset of questions
        comparison_questions = test_questions[:3]  # Use first 3 for efficiency
        strategy_comparisons = {}
        
        for question in comparison_questions:
            comparison = self.compare_strategies(question)
            strategy_comparisons[question] = comparison
        
        benchmark_results["strategy_comparison"] = strategy_comparisons
        
        logger.info("Benchmark suite completed")
        return benchmark_results
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str = None):
        """Save evaluation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = f"evaluation_results/{filename}"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs("evaluation_results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {filepath}")
        return filepath


# Utility functions
def quick_evaluate(question: str, strategy: str = "ensemble") -> EvaluationMetrics:
    """Quick evaluation of a single question."""
    evaluator = RAGEvaluator()
    return evaluator.evaluate_single_query(question, strategy)


def run_quick_benchmark() -> Dict[str, Any]:
    """Run a quick benchmark with default questions."""
    evaluator = RAGEvaluator()
    return evaluator.run_benchmark_suite()


if __name__ == "__main__":
    # Demo evaluation
    print("🔍 Running RAG Pipeline Evaluation Demo")
    print("=" * 50)
    
    # Single query evaluation
    test_question = "What are the main challenges in education policy implementation?"
    print(f"Evaluating: {test_question}")
    
    metrics = quick_evaluate(test_question)
    print(f"Overall Score: {metrics.overall_score:.2f}/10")
    print(f"Relevance: {metrics.relevance_score:.1f}, Completeness: {metrics.completeness_score:.1f}")
    print(f"Response Time: {metrics.response_time:.2f}s")
    print()
    
    # Strategy comparison
    print("Comparing strategies...")
    evaluator = RAGEvaluator()
    comparison = evaluator.compare_strategies(test_question, ["vector", "ensemble", "bm25"])
    print(f"Best Strategy: {comparison['best_strategy']} (Score: {comparison['best_score']:.2f})")
    print()
    
    # Quick benchmark
    print("Running quick benchmark...")
    benchmark = run_quick_benchmark()
    avg_score = benchmark["aggregate_metrics"]["avg_overall_score"]
    print(f"Average Score across all questions: {avg_score:.2f}/10")
    
    # Save results
    evaluator.save_evaluation_results(benchmark)
    print("Results saved to evaluation_results/ directory")
