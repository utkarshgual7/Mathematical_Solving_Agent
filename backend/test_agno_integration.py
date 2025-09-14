#!/usr/bin/env python3
"""
Test script for Agno integration with the Mathematical Routing Agent

This script tests:
1. Agno knowledge base initialization
2. PDF processing and storage
3. Mathematical problem solving
4. Web search integration
5. Vector similarity search
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.vector_service import AgnoMathKnowledgeBase, load_sample_math_dataset
from app.services.pdf_processing_service import MathPDFProcessor
from app.services.mcp_service import WebSearchMCP
from app.agents.math_agent import MathRoutingAgent
from app.core.config import settings

class AgnoIntegrationTester:
    """Test suite for Agno integration"""
    
    def __init__(self):
        self.test_results = []
        self.knowledge_base = None
        self.math_agent = None
        self.web_search = None
        
    def log_test(self, test_name: str, success: bool, message: str = "", details: dict = None):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    async def test_knowledge_base_initialization(self):
        """Test Agno knowledge base initialization"""
        try:
            self.knowledge_base = AgnoMathKnowledgeBase()
            
            # Check if knowledge base components are initialized
            has_embedder = hasattr(self.knowledge_base, 'embedder') and self.knowledge_base.embedder is not None
            has_vector_db = hasattr(self.knowledge_base, 'vector_db') and self.knowledge_base.vector_db is not None
            has_pdf_processor = hasattr(self.knowledge_base, 'pdf_processor') and self.knowledge_base.pdf_processor is not None
            
            if has_embedder and has_vector_db and has_pdf_processor:
                stats = self.knowledge_base.get_stats()
                self.log_test(
                    "Knowledge Base Initialization",
                    True,
                    "Agno knowledge base initialized successfully",
                    stats
                )
            else:
                self.log_test(
                    "Knowledge Base Initialization",
                    False,
                    "Missing required components",
                    {
                        "has_embedder": has_embedder,
                        "has_vector_db": has_vector_db,
                        "has_pdf_processor": has_pdf_processor
                    }
                )
                
        except Exception as e:
            self.log_test(
                "Knowledge Base Initialization",
                False,
                f"Error initializing knowledge base: {str(e)}"
            )
    
    async def test_sample_data_loading(self):
        """Test loading sample mathematical problems"""
        try:
            # Load sample dataset
            sample_kb = await load_sample_math_dataset()
            
            if sample_kb:
                stats = sample_kb.get_stats()
                self.log_test(
                    "Sample Data Loading",
                    True,
                    "Sample mathematical problems loaded successfully",
                    stats
                )
                
                # Update our knowledge base reference
                self.knowledge_base = sample_kb
            else:
                self.log_test(
                    "Sample Data Loading",
                    False,
                    "Failed to load sample dataset"
                )
                
        except Exception as e:
            self.log_test(
                "Sample Data Loading",
                False,
                f"Error loading sample data: {str(e)}"
            )
    
    async def test_vector_search(self):
        """Test vector similarity search"""
        try:
            if not self.knowledge_base:
                self.log_test(
                    "Vector Search",
                    False,
                    "Knowledge base not available for testing"
                )
                return
            
            # Test search queries
            test_queries = [
                "derivative of polynomial",
                "quadratic equation solution",
                "integration techniques",
                "calculus problems"
            ]
            
            search_results = {}
            
            for query in test_queries:
                results = await self.knowledge_base.search_similar_problems(query, limit=3)
                search_results[query] = {
                    "count": len(results),
                    "results": results[:2] if results else []  # Show first 2 results
                }
            
            # Check if we got meaningful results
            total_results = sum(result["count"] for result in search_results.values())
            
            if total_results > 0:
                self.log_test(
                    "Vector Search",
                    True,
                    f"Vector search working - found {total_results} total results",
                    search_results
                )
            else:
                self.log_test(
                    "Vector Search",
                    False,
                    "No search results found",
                    search_results
                )
                
        except Exception as e:
            self.log_test(
                "Vector Search",
                False,
                f"Error in vector search: {str(e)}"
            )
    
    async def test_web_search_integration(self):
        """Test Agno web search integration"""
        try:
            self.web_search = WebSearchMCP()
            
            # Test search query
            test_query = "how to solve quadratic equations step by step"
            
            # Test basic search
            search_results = await self.web_search.search_math_content(test_query, max_results=3)
            
            if search_results:
                self.log_test(
                    "Web Search Integration",
                    True,
                    f"Web search working - found {len(search_results)} results",
                    {
                        "query": test_query,
                        "results_count": len(search_results),
                        "sample_result": search_results[0] if search_results else None
                    }
                )
            else:
                self.log_test(
                    "Web Search Integration",
                    False,
                    "No web search results found",
                    {"query": test_query}
                )
                
        except Exception as e:
            self.log_test(
                "Web Search Integration",
                False,
                f"Error in web search: {str(e)}"
            )
    
    async def test_math_agent_initialization(self):
        """Test mathematical routing agent initialization"""
        try:
            # Check if we have required API keys
            if not has_gemini:
                self.log_test(
                    "Math Agent Initialization",
                    False,
                    "No API keys available for LLM models",
                    {
                        "has_gemini_key": has_gemini,
                        "note": "Set GEMINI_API_KEY in .env file"
                    }
                )
                return
            
            # Initialize math agent
            self.math_agent = MathRoutingAgent(web_search=self.web_search)
            
            # Check agent components
            has_solver = hasattr(self.math_agent, 'solver_agent') and self.math_agent.solver_agent is not None
            has_knowledge_base = hasattr(self.math_agent, 'knowledge_base') and self.math_agent.knowledge_base is not None
            has_tools = hasattr(self.math_agent, 'math_tools') and self.math_agent.math_tools is not None
            
            if has_solver and has_knowledge_base and has_tools:
                self.log_test(
                    "Math Agent Initialization",
                    True,
                    "Mathematical routing agent initialized successfully",
                    {
                        "has_solver_agent": has_solver,
                        "has_knowledge_base": has_knowledge_base,
                        "has_math_tools": has_tools,
                        "api_keys_available": {"gemini": has_gemini}
                    }
                )
            else:
                self.log_test(
                    "Math Agent Initialization",
                    False,
                    "Missing required agent components",
                    {
                        "has_solver_agent": has_solver,
                        "has_knowledge_base": has_knowledge_base,
                        "has_math_tools": has_tools
                    }
                )
                
        except Exception as e:
            self.log_test(
                "Math Agent Initialization",
                False,
                f"Error initializing math agent: {str(e)}"
            )
    
    async def test_end_to_end_problem_solving(self):
        """Test end-to-end mathematical problem solving"""
        try:
            if not self.math_agent:
                self.log_test(
                    "End-to-End Problem Solving",
                    False,
                    "Math agent not available for testing"
                )
                return
            
            # Test problem
            test_problem = "Find the derivative of f(x) = 3x^2 + 2x - 1"
            
            # Solve the problem
            result = await self.math_agent.solve_problem(test_problem)
            
            if result and result.get('solution'):
                self.log_test(
                    "End-to-End Problem Solving",
                    True,
                    "Successfully solved mathematical problem",
                    {
                        "problem": test_problem,
                        "solution_preview": result['solution'][:200] + "..." if len(result['solution']) > 200 else result['solution'],
                        "confidence": result.get('confidence', 'N/A'),
                        "reasoning_steps": len(result.get('reasoning_steps', [])),
                        "sources_used": len(result.get('sources', []))
                    }
                )
            else:
                self.log_test(
                    "End-to-End Problem Solving",
                    False,
                    "Failed to solve mathematical problem",
                    {"problem": test_problem, "result": result}
                )
                
        except Exception as e:
            self.log_test(
                "End-to-End Problem Solving",
                False,
                f"Error in problem solving: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 Starting Agno Integration Tests...\n")
        
        # Run tests in sequence
        await self.test_knowledge_base_initialization()
        await self.test_sample_data_loading()
        await self.test_vector_search()
        await self.test_web_search_integration()
        await self.test_math_agent_initialization()
        await self.test_end_to_end_problem_solving()
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Test Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Save detailed results
        results_file = Path("agno_integration_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100,
                    "test_timestamp": datetime.now().isoformat()
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        if failed_tests > 0:
            print("\n⚠️  Some tests failed. Check the detailed results for more information.")
            print("Common issues:")
            print("- Missing API keys (GEMINI_API_KEY)")
            print("- Network connectivity for web search")
            print("- Missing dependencies (run: pip install -r requirements.txt)")
        else:
            print("\n🎉 All tests passed! Agno integration is working correctly.")

async def main():
    """Main test runner"""
    tester = AgnoIntegrationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())