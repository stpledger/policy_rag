#!/usr/bin/env python3
"""
Benchmark Script for Education Policy RAG System

This script runs comprehensive performance benchmarks across all
retrieval strategies and generates detailed performance reports.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.rag_pipeline import EnhancedRAGPipeline
from src.evaluation.evaluation import RAGEvaluator

def main():
    """Run comprehensive benchmarks."""
    print("📊 Education Policy RAG System Benchmark")
    print("=" * 50)
    
    try:
        # Initialize system
        pipeline = EnhancedRAGPipeline()
        evaluator = RAGEvaluator(pipeline=pipeline)
        
        # Define test questions
        test_questions = [
            "What are effective reading intervention strategies for struggling students?",
            "How do school choice programs impact student outcomes?",
            "What role does teacher professional development play in student achievement?",
            "How can technology be effectively integrated into classroom instruction?",
            "What are the key challenges in education policy implementation?",
        ]
        
        print(f"Running benchmark with {len(test_questions)} test questions...")
        
        # Run benchmark
        results = evaluator.run_benchmark_suite(test_questions)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Benchmark complete! Results saved to {filename}")
        print(f"📈 Overall performance: {results.get('average_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
