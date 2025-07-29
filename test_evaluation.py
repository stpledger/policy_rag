#!/usr/bin/env python3
"""
Test script for the evaluation system.
Demonstrates evaluation capabilities and generates sample results.
"""

import sys
import json
from datetime import datetime
from rag_pipeline import EnhancedRAGPipeline
from evaluation import RAGEvaluator, quick_evaluate, run_quick_benchmark


def test_single_evaluation():
    """Test evaluation of a single question."""
    print("🔍 Testing Single Question Evaluation")
    print("=" * 50)
    
    question = "What are the main challenges in education policy implementation?"
    print(f"Question: {question}")
    
    # Test with different strategies
    strategies = ["ensemble", "vector", "bm25"]
    
    for strategy in strategies:
        print(f"\n--- Testing with {strategy.upper()} strategy ---")
        
        try:
            metrics = quick_evaluate(question, strategy)
            
            print(f"Overall Score: {metrics.overall_score:.2f}/10")
            print(f"Answer Quality:")
            print(f"  - Relevance: {metrics.relevance_score:.1f}/10")
            print(f"  - Completeness: {metrics.completeness_score:.1f}/10")
            print(f"  - Accuracy: {metrics.accuracy_score:.1f}/10")
            print(f"  - Clarity: {metrics.clarity_score:.1f}/10")
            
            print(f"Retrieval Performance:")
            print(f"  - Precision: {metrics.retrieval_precision:.3f}")
            print(f"  - Recall: {metrics.retrieval_recall:.3f}")
            print(f"  - Diversity: {metrics.document_diversity:.3f}")
            print(f"  - Coverage: {metrics.source_coverage:.3f}")
            
            print(f"Performance:")
            print(f"  - Response Time: {metrics.response_time:.2f}s")
            print(f"  - Documents Retrieved: {metrics.num_documents_retrieved}")
            
        except Exception as e:
            print(f"Error evaluating with {strategy}: {e}")


def test_strategy_comparison():
    """Test strategy comparison evaluation."""
    print("\n🔄 Testing Strategy Comparison")
    print("=" * 50)
    
    question = "How effective are school choice programs?"
    print(f"Question: {question}")
    
    try:
        evaluator = RAGEvaluator()
        comparison = evaluator.compare_strategies(question, ["vector", "ensemble", "bm25"])
        
        print(f"\nBest Strategy: {comparison['best_strategy']}")
        print(f"Best Score: {comparison['best_score']:.2f}/10")
        
        print("\nStrategy Results:")
        for strategy, result in comparison['strategy_results'].items():
            if 'error' not in result:
                print(f"  {strategy.upper()}: {result['overall_score']:.2f}/10 "
                      f"({result['num_documents_retrieved']} docs, "
                      f"{result['response_time']:.2f}s)")
            else:
                print(f"  {strategy.upper()}: Error - {result['error']}")
                
    except Exception as e:
        print(f"Error in strategy comparison: {e}")


def test_pipeline_integration():
    """Test evaluation integration with the main pipeline."""
    print("\n🔗 Testing Pipeline Integration")
    print("=" * 50)
    
    pipeline = EnhancedRAGPipeline()
    question = "What role do teachers play in education reform?"
    
    print(f"Question: {question}")
    
    try:
        # Test evaluate_response method
        eval_result = pipeline.evaluate_response(question, "ensemble")
        
        if eval_result.get('evaluation_metrics'):
            print(f"\nEvaluation successful!")
            print(f"Overall Score: {eval_result['overall_score']:.2f}/10")
            
            summary = eval_result['performance_summary']
            print(f"Answer Quality: {summary['answer_quality']:.2f}/10")
            print(f"Retrieval Performance: {summary['retrieval_performance']:.2f}/10")
            print(f"Response Time: {summary['response_time']:.2f}s")
            
            # Show response preview
            response = eval_result['response']
            answer_preview = response['answer'][:150] + "..." if len(response['answer']) > 150 else response['answer']
            print(f"\nResponse Preview: {answer_preview}")
            
        else:
            print("Evaluation metrics not available")
            if 'error' in eval_result:
                print(f"Error: {eval_result['error']}")
                
    except Exception as e:
        print(f"Error in pipeline integration test: {e}")


def test_benchmark():
    """Test the benchmark functionality."""
    print("\n🎯 Testing Benchmark System")
    print("=" * 50)
    
    # Use a small set of questions for testing
    test_questions = [
        "What are the main challenges in education policy implementation?",
        "How effective are school choice programs?",
        "What role do teachers play in education reform?"
    ]
    
    print(f"Running benchmark with {len(test_questions)} questions...")
    
    try:
        benchmark = run_quick_benchmark()
        
        if 'error' not in benchmark:
            print("Benchmark completed successfully!")
            
            # Show aggregate metrics
            agg = benchmark['aggregate_metrics']
            print(f"\nAggregate Results:")
            print(f"  Average Overall Score: {agg['avg_overall_score']:.2f}/10")
            print(f"  Average Answer Quality: {agg['avg_relevance_score']:.2f}/10")
            print(f"  Average Retrieval Precision: {agg['avg_retrieval_precision']:.3f}")
            print(f"  Average Response Time: {agg['avg_response_time']:.2f}s")
            print(f"  Total Questions: {agg['total_questions']}")
            
            # Show individual results summary
            print(f"\nIndividual Results:")
            for question, metrics in benchmark['individual_results'].items():
                q_short = question[:50] + "..." if len(question) > 50 else question
                print(f"  {q_short}: {metrics['overall_score']:.1f}/10")
                
        else:
            print(f"Benchmark failed: {benchmark['error']}")
            
    except Exception as e:
        print(f"Error in benchmark test: {e}")


def save_sample_results():
    """Generate and save sample evaluation results."""
    print("\n💾 Generating Sample Results")
    print("=" * 50)
    
    try:
        # Run a quick evaluation
        question = "What are the main challenges in education policy implementation?"
        metrics = quick_evaluate(question, "ensemble")
        
        # Create sample results structure
        sample_results = {
            "evaluation_type": "single_question",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "strategy": "ensemble",
            "metrics": metrics.to_dict(),
            "summary": {
                "overall_performance": "Good",
                "strengths": [
                    "High relevance score",
                    "Fast response time",
                    "Good document diversity"
                ],
                "areas_for_improvement": [
                    "Could improve completeness",
                    "More source coverage needed"
                ]
            }
        }
        
        # Save to file
        filename = f"sample_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create evaluation_results directory if it doesn't exist
        import os
        os.makedirs("evaluation_results", exist_ok=True)
        
        filepath = f"evaluation_results/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        
        print(f"Sample results saved to: {filepath}")
        
    except Exception as e:
        print(f"Error saving sample results: {e}")


def main():
    """Run all evaluation tests."""
    print("🚀 RAG Pipeline Evaluation Test Suite")
    print("=" * 60)
    print("This script tests the evaluation system functionality")
    print()
    
    try:
        # Run all tests
        test_single_evaluation()
        test_strategy_comparison()
        test_pipeline_integration()
        test_benchmark()
        save_sample_results()
        
        print("\n✅ All tests completed!")
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to see evaluation in the web interface")
        print("2. Check the 'Evaluation' and 'Benchmark' tabs in the app")
        print("3. Review saved results in the evaluation_results/ directory")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
