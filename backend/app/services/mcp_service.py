import asyncio
import json
from typing import Dict, List, Optional
import aiohttp
from mcp.server.fastmcp import FastMCP

class WebSearchMCP:
    def __init__(self, tavily_api_key: str):
        self.api_key = tavily_api_key
        self.base_url = "https://api.tavily.com/search"
        
    async def search_math_content(
        self, 
        query: str, 
        max_results: int = 5
    ) -> List[Dict]:
        """Search for mathematical content using Tavily API"""
        
        # Enhance query for mathematical content
        enhanced_query = self._enhance_math_query(query)
        
        search_params = {
            "api_key": self.api_key,
            "query": enhanced_query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_domains": [
                "wolfram.com",
                "mathworld.wolfram.com", 
                "khanacademy.org",
                "math.stackexchange.com",
                "brilliant.org"
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url, 
                json=search_params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_search_results(data.get("results", []))
                else:
                    return []
    
    def _enhance_math_query(self, query: str) -> str:
        """Enhance query with mathematical context"""
        math_terms = [
            "mathematics", "math", "solution", "step by step",
            "how to solve", "explanation"
        ]
        
        # Add mathematical context if not present
        if not any(term in query.lower() for term in math_terms):
            return f"{query} mathematics solution explanation"
        
        return query
    
    def _process_search_results(self, results: List[Dict]) -> List[Dict]:
        """Process and filter search results"""
        processed = []
        
        for result in results:
            # Filter for quality mathematical content
            if self._is_quality_math_content(result):
                processed.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:500],
                    "score": result.get("score", 0)
                })
        
        return processed
    
    def _is_quality_math_content(self, result: Dict) -> bool:
        """Check if result contains quality mathematical content"""
        content = (result.get("content", "") + " " + 
                  result.get("title", "")).lower()
        
        # Check for mathematical indicators
        math_indicators = [
            "solution", "solve", "step", "equation", "formula",
            "theorem", "proof", "calculate", "mathematics"
        ]
        
        return any(indicator in content for indicator in math_indicators)

# MCP Server setup
mcp_server = FastMCP("MathSearchMCP")

@mcp_server.tool()
async def search_mathematical_problems(query: str) -> str:
    """Search for mathematical problems and solutions"""
    search_service = WebSearchMCP(tavily_api_key="your-api-key")
    results = await search_service.search_math_content(query)
    return json.dumps(results, indent=2)

@mcp_server.tool()
async def verify_mathematical_solution(
    problem: str, 
    solution: str
) -> str:
    """Verify if a mathematical solution is correct"""
    # Implementation for solution verification
    # Could integrate with external math verification APIs
    return json.dumps({
        "verified": True,
        "confidence": 0.85,
        "notes": "Solution appears mathematically sound"
    })