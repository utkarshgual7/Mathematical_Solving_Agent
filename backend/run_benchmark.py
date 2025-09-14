#!/usr/bin/env python3
"""
Script to run JEE Benchmark Evaluation for the Mathematical Routing Agent

This script executes the benchmark evaluation using the JEEBenchmarkEvaluator
against the PhysicsWallahAI/JEE-Main-2025-Math dataset.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.agents.math_agent import MathRoutingAgent
from app.utils.evaluation import JEEBenchmarkEvaluator
from app.services.vector_service import MathKnowledgeBase
from app.services.mcp_service import WebSearchMCP
from app.core.config import settings

async def run_benchmark():
    """Run the JEE benchmark evaluation"""
    print("🚀 Starting JEE Benchmark Evaluation...")
    
    try:
        # Initialize services
        print("🔧 Initializing services...")
        
        # Initialize knowledge base
        knowledge_base = MathKnowledgeBase()
        
        # Initialize web search (this would require a Tavily API key)
        web_search = WebSearchMCP(settings.TAVILY_API_KEY if hasattr(settings, 'TAVILY_API_KEY') else None)
        
        # Initialize math agent
        math_agent = MathRoutingAgent(web_search)
        
        # Initialize evaluator
        evaluator = JEEBenchmarkEvaluator(math_agent)
        
        # Run evaluation
        print("🧪 Running evaluation on JEE Main 2025 Math dataset...")
        results = await evaluator.run_evaluation("PhysicsWallahAI/JEE-Main-2025-Math")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Total Problems: {results['total_problems']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Average Confidence: {results['average_confidence']:.2f}")
        
        # Save detailed results
        results_file = "jee_benchmark_results_detailed.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to {results_file}")
        
        # Save summary
        summary = {
            "total_problems": results["total_problems"],
            "correct_answers": results["correct_answers"],
            "accuracy": results["accuracy"],
            "average_confidence": results["average_confidence"]
        }
        
        summary_file = "jee_benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📋 Summary saved to {summary_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point"""
    print("🎓 JEE Benchmark Evaluation Script")
    print("="*50)
    
    # Check if required environment variables are set
    if hasattr(settings, 'TAVILY_API_KEY') and settings.TAVILY_API_KEY:
        print("✅ Tavily API key found")
    else:
        print("⚠️  Warning: Tavily API key not found. Web search functionality will be limited.")
        print("   Set TAVILY_API_KEY in your environment for full functionality.")
    
    # Run the benchmark
    results = asyncio.run(run_benchmark())
    
    if results:
        print("\n✅ Benchmark evaluation completed successfully!")
        return 0
    else:
        print("\n❌ Benchmark evaluation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())