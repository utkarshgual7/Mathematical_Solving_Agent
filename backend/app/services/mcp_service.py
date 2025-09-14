import asyncio
import json
from typing import Dict, List, Optional, Any
import aiohttp

# Agno imports for enhanced web search
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini

from app.core.config import settings

class WebSearchMCP:
    """Enhanced web search service using both Tavily and Agno DuckDuckGo tools"""
    
    def __init__(self, tavily_api_key: str = None):
        self.api_key = tavily_api_key
        self.base_url = "https://api.tavily.com/search"
        
        # Initialize Agno DuckDuckGo tools
        self.ddg_tools = DuckDuckGoTools()
        
        # Setup Agno search agent for complex queries
        self._setup_search_agent()
        
    def _setup_search_agent(self):
        """Setup Agno agent for intelligent web search"""
        try:
            # Choose model based on available API keys
            if settings.OPENAI_API_KEY:
                model = OpenAIChat(id="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
            elif settings.GEMINI_API_KEY:
                model = Gemini(id="gemini-2.0-flash-exp", api_key=settings.GEMINI_API_KEY)
            else:
                model = None
            
            if model:
                self.search_agent = Agent(
                    name="Mathematical Web Search Agent",
                    model=model,
                    description="You are an expert at finding relevant mathematical content on the web.",
                    instructions=[
                        "Search for high-quality mathematical content and explanations",
                        "Focus on educational resources, step-by-step solutions, and authoritative sources",
                        "Prioritize content from Khan Academy, Wolfram, Math Stack Exchange, and academic sources",
                        "Extract key mathematical concepts and solution methods",
                        "Provide clear, concise summaries of found content"
                    ],
                    tools=[self.ddg_tools],
                    markdown=True,
                    show_tool_calls=False
                )
            else:
                self.search_agent = None
        except Exception as e:
            print(f"Error setting up search agent: {e}")
            self.search_agent = None
    
    async def search_math_content(
        self, 
        query: str, 
        max_results: int = 5,
        use_agno: bool = True
    ) -> List[Dict]:
        """Search for mathematical content using Agno DuckDuckGo tools and/or Tavily API"""
        
        results = []
        
        # Try Agno DuckDuckGo search first (free and often effective)
        if use_agno and self.search_agent:
            try:
                agno_results = await self._search_with_agno(query, max_results)
                results.extend(agno_results)
            except Exception as e:
                print(f"Agno search failed: {e}")
        
        # Fallback to Tavily if needed and API key available
        if len(results) < max_results and self.api_key:
            try:
                tavily_results = await self._search_with_tavily(query, max_results - len(results))
                results.extend(tavily_results)
            except Exception as e:
                print(f"Tavily search failed: {e}")
        
        # If no results from either, try basic DuckDuckGo
        if not results:
            try:
                basic_results = await self._basic_ddg_search(query, max_results)
                results.extend(basic_results)
            except Exception as e:
                print(f"Basic DuckDuckGo search failed: {e}")
        
        return results[:max_results]
    
    async def _search_with_agno(self, query: str, max_results: int) -> List[Dict]:
        """Search using Agno agent with DuckDuckGo tools"""
        try:
            enhanced_query = self._enhance_math_query(query)
            
            # Use Agno agent to perform intelligent search
            search_prompt = f"""
            Search for mathematical content related to: {enhanced_query}
            
            Focus on finding:
            1. Step-by-step solutions and explanations
            2. Educational resources and tutorials
            3. Mathematical concepts and definitions
            4. Practice problems and examples
            
            Provide a summary of the most relevant findings with sources.
            """
            
            response = await self.search_agent.arun(search_prompt)
            
            # Extract search results from agent response
            # This is a simplified extraction - in practice, you'd parse the agent's tool calls
            return self._parse_agno_response(response, max_results)
            
        except Exception as e:
            print(f"Error in Agno search: {e}")
            return []
    
    async def _search_with_tavily(self, query: str, max_results: int) -> List[Dict]:
        """Search using Tavily API"""
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
                "brilliant.org",
                "mathpages.com",
                "cut-the-knot.org"
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
    
    async def _basic_ddg_search(self, query: str, max_results: int) -> List[Dict]:
        """Basic DuckDuckGo search as fallback"""
        try:
            enhanced_query = self._enhance_math_query(query)
            
            # Use DuckDuckGo tools directly
            search_results = self.ddg_tools.search(enhanced_query, max_results=max_results)
            
            # Format results
            formatted_results = []
            if isinstance(search_results, list):
                for result in search_results:
                    if isinstance(result, dict):
                        formatted_results.append({
                            "title": result.get("title", ""),
                            "url": result.get("href", result.get("url", "")),
                            "snippet": result.get("body", result.get("snippet", ""))[:500],
                            "score": 0.7  # Default score for basic search
                        })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in basic DuckDuckGo search: {e}")
            return []
    
    def _parse_agno_response(self, response: str, max_results: int) -> List[Dict]:
        """Parse Agno agent response to extract search results"""
        # This is a simplified parser - in practice, you'd extract from tool calls
        results = []
        
        # Look for URLs and content in the response
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, response)
        
        # Create basic results from found URLs
        for i, url in enumerate(urls[:max_results]):
            results.append({
                "title": f"Mathematical Resource {i+1}",
                "url": url,
                "snippet": "Content found through Agno intelligent search",
                "score": 0.8
            })
        
        return results
    
    def _enhance_math_query(self, query: str) -> str:
        """Enhance query with mathematical context"""
        math_terms = [
            "mathematics", "math", "solution", "step by step",
            "how to solve", "explanation", "tutorial", "example"
        ]
        
        # Add mathematical context if not present
        if not any(term in query.lower() for term in math_terms):
            return f"{query} mathematics solution step by step explanation"
        
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
                    "snippet": result.get("content", result.get("snippet", ""))[:500],
                    "score": result.get("score", 0.7),
                    "source": "tavily_search"
                })
        
        return processed
    
    def _is_quality_math_content(self, result: Dict) -> bool:
        """Check if result contains quality mathematical content"""
        content = (result.get("content", "") + " " + 
                  result.get("title", "") + " " +
                  result.get("snippet", "")).lower()
        
        # Check for mathematical indicators
        math_indicators = [
            "solution", "solve", "step", "equation", "formula",
            "theorem", "proof", "calculate", "mathematics",
            "derivative", "integral", "algebra", "geometry",
            "trigonometry", "calculus", "statistics", "probability",
            "linear", "quadratic", "polynomial", "function",
            "graph", "plot", "example", "practice", "tutorial"
        ]
        
        # Quality domains
        quality_domains = [
            "khanacademy", "wolfram", "mathworld", "brilliant",
            "stackexchange", "mathpages", "cut-the-knot",
            "purplemath", "mathisfun", "coursera", "edx"
        ]
        
        url = result.get("url", "").lower()
        
        # Check for math indicators or quality domains
        has_math_content = any(indicator in content for indicator in math_indicators)
        is_quality_domain = any(domain in url for domain in quality_domains)
        
        return has_math_content or is_quality_domain

    async def search_and_summarize(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search for mathematical content and provide AI-powered summary"""
        try:
            # Get search results
            results = await self.search_math_content(query, max_results)
            
            if not results or not self.search_agent:
                return {
                    "query": query,
                    "results": results,
                    "summary": "No results found or search agent unavailable"
                }
            
            # Create summary using Agno agent
            summary_prompt = f"""
            Based on the following search results for the query "{query}", provide a comprehensive summary:
            
            {json.dumps(results, indent=2)}
            
            Please provide:
            1. A concise overview of the mathematical topic
            2. Key concepts and formulas mentioned
            3. Step-by-step approach if applicable
            4. Most relevant sources for further learning
            """
            
            summary = await self.search_agent.arun(summary_prompt)
            
            return {
                "query": query,
                "results": results,
                "summary": summary,
                "total_results": len(results)
            }
            
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "summary": f"Error generating summary: {str(e)}",
                "error": str(e)
            }

# MCP Server functionality can be added later when FastMCP is available
# For now, the WebSearchMCP class provides the core functionality

# Example usage:
# search_service = WebSearchMCP()
# results = await search_service.search_math_content("quadratic equations", max_results=5)
# summary = await search_service.search_and_summarize("calculus derivatives", max_results=3)