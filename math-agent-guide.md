# Mathematical Routing Agent - Complete Implementation Guide

## 1. Project Architecture Overview

### Core Components Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Chat Interface │  │ Feedback UI │  │ Admin Panel │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                          HTTP/WebSocket
                              │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ API Gateway │  │ Auth System │  │ Guardrails  │         │
│  │ Routing     │  │             │  │ Layer       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Math Agent  │  │ Knowledge   │  │ Web Search  │         │
│  │ (DSPy)      │  │ Base RAG    │  │ MCP Server  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Human-Loop  │  │ Feedback    │  │ Evaluation  │         │
│  │ Manager     │  │ Processor   │  │ Engine      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Vector DB   │  │ PostgreSQL  │  │ Redis Cache │         │
│  │ (Qdrant)    │  │ (Metadata)  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 2. Implementation Roadmap

### Phase 1: Backend Core Setup
1. FastAPI application structure
2. Database models and migrations  
3. Vector database integration (Qdrant)
4. Basic API endpoints

### Phase 2: AI Components
1. AI Gateway with guardrails implementation
2. Mathematical knowledge base creation
3. DSPy-based Math Agent development
4. MCP server for web search

### Phase 3: Advanced Features
1. Human-in-the-loop system
2. Feedback processing and learning
3. JEE benchmark evaluation

### Phase 4: Frontend & Integration
1. React application development
2. API integration
3. Real-time feedback interface

## 3. Detailed Implementation

### 3.1 FastAPI Backend Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── math_agent.py
│   │   │   ├── feedback.py
│   │   │   └── knowledge.py
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── guardrails.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── math_agent.py
│   │   ├── knowledge_retriever.py
│   │   └── web_search.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_service.py
│   │   ├── feedback_service.py
│   │   └── mcp_service.py
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py
│       └── evaluation.py
├── requirements.txt
└── docker-compose.yml
```

### 3.2 Core Configuration (app/core/config.py)

```python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Math Routing Agent"
    
    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "math_agent"
    
    # Vector Database Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # LLM Settings
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # MCP Settings
    TAVILY_API_KEY: str = ""
    
    # Security
    SECRET_KEY: str = "your-secret-key"
    
    # Guardrails
    ENABLE_GUARDRAILS: bool = True
    MAX_TOKENS_PER_MINUTE: int = 10000
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 3.3 AI Gateway with Guardrails (app/core/guardrails.py)

```python
import re
import time
from typing import Dict, List, Optional
from fastapi import HTTPException
from pydantic import BaseModel

class GuardrailsConfig(BaseModel):
    blocked_topics: List[str] = ["politics", "religion", "violence"]
    max_tokens: int = 4000
    educational_keywords: List[str] = [
        "math", "mathematics", "equation", "solve", "calculate", 
        "algebra", "geometry", "calculus", "statistics", "probability"
    ]

class InputGuardrails:
    def __init__(self, config: GuardrailsConfig):
        self.config = config
        self.request_cache: Dict[str, List[float]] = {}
        
    async def validate_input(self, prompt: str, user_id: str) -> Dict[str, any]:
        """Comprehensive input validation with educational focus"""
        
        # Rate limiting check
        await self._check_rate_limit(user_id)
        
        # Content moderation
        if not await self._is_educational_content(prompt):
            raise HTTPException(
                status_code=400, 
                detail="Content must be mathematics-focused"
            )
        
        # Blocked content check
        if await self._contains_blocked_content(prompt):
            raise HTTPException(
                status_code=400,
                detail="Content contains blocked topics"
            )
        
        # Token count validation
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
        if estimated_tokens > self.config.max_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Input too long. Max {self.config.max_tokens} tokens"
            )
            
        return {"status": "approved", "estimated_tokens": estimated_tokens}
    
    async def _check_rate_limit(self, user_id: str):
        """Rate limiting per user"""
        current_time = time.time()
        if user_id not in self.request_cache:
            self.request_cache[user_id] = []
        
        # Remove requests older than 1 minute
        self.request_cache[user_id] = [
            req_time for req_time in self.request_cache[user_id]
            if current_time - req_time < 60
        ]
        
        if len(self.request_cache[user_id]) > 20:  # Max 20 requests per minute
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        self.request_cache[user_id].append(current_time)
    
    async def _is_educational_content(self, prompt: str) -> bool:
        """Check if content is mathematics/education focused"""
        prompt_lower = prompt.lower()
        
        # Check for mathematical keywords
        math_keywords_found = any(
            keyword in prompt_lower 
            for keyword in self.config.educational_keywords
        )
        
        # Check for mathematical symbols/patterns
        math_patterns = [
            r'\d+[\+\-\*/]\d+',  # Basic arithmetic
            r'[a-z]\s*=\s*\d+',  # Variable assignments
            r'\\[a-zA-Z]+\{.*?\}',  # LaTeX commands
            r'\b(solve|find|calculate|compute|determine)\b'  # Action words
        ]
        
        pattern_found = any(
            re.search(pattern, prompt_lower) 
            for pattern in math_patterns
        )
        
        return math_keywords_found or pattern_found
    
    async def _contains_blocked_content(self, prompt: str) -> bool:
        """Check for blocked topics"""
        prompt_lower = prompt.lower()
        return any(
            blocked in prompt_lower 
            for blocked in self.config.blocked_topics
        )

class OutputGuardrails:
    def __init__(self):
        self.inappropriate_patterns = [
            r'I cannot help with that',
            r'inappropriate',
            r'harmful',
        ]
    
    async def validate_output(self, response: str) -> Dict[str, any]:
        """Validate LLM output for appropriateness"""
        
        # Check for mathematical content
        if not self._contains_mathematical_content(response):
            return {
                "status": "flagged",
                "reason": "Response lacks mathematical content"
            }
        
        # Check response length
        if len(response.split()) < 10:
            return {
                "status": "flagged",
                "reason": "Response too short"
            }
        
        return {"status": "approved"}
    
    def _contains_mathematical_content(self, response: str) -> bool:
        """Check if response contains mathematical content"""
        math_indicators = [
            r'\d+',  # Numbers
            r'[=\+\-\*/]',  # Mathematical operators
            r'\b(step|solve|answer|solution|equation)\b',  # Mathematical terms
        ]
        
        return any(
            re.search(pattern, response.lower())
            for pattern in math_indicators
        )
```

### 3.4 Mathematical Knowledge Base Setup (app/services/vector_service.py)

```python
import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json

class MathKnowledgeBase:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "math_knowledge"
        self.setup_collection()
    
    def setup_collection(self):
        """Create and configure Qdrant collection"""
        collections = self.client.get_collections().collections
        collection_exists = any(
            col.name == self.collection_name for col in collections
        )
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=models.Distance.COSINE
                )
            )
    
    async def add_math_problems(self, problems: List[Dict]):
        """Add mathematical problems to knowledge base"""
        points = []
        
        for i, problem in enumerate(problems):
            # Create text representation
            text = self._create_problem_text(problem)
            
            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()
            
            # Create point
            point = models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "question": problem["question"],
                    "solution": problem["solution"],
                    "topic": problem.get("topic", "general"),
                    "difficulty": problem.get("difficulty", "medium"),
                    "source": problem.get("source", "unknown")
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    async def search_similar_problems(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Search for similar problems in knowledge base"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for scored_point in search_result:
            results.append({
                "score": scored_point.score,
                "question": scored_point.payload["question"],
                "solution": scored_point.payload["solution"],
                "topic": scored_point.payload["topic"],
                "difficulty": scored_point.payload["difficulty"]
            })
        
        return results
    
    def _create_problem_text(self, problem: Dict) -> str:
        """Create searchable text representation of problem"""
        components = [
            problem["question"],
            problem.get("topic", ""),
            problem.get("difficulty", "")
        ]
        return " ".join(filter(None, components))

# Sample dataset loading
async def load_jee_dataset():
    """Load JEE Main 2025 Math dataset"""
    # This would integrate with the PhysicsWallahAI/JEE-Main-2025-Math dataset
    sample_problems = [
        {
            "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "solution": "f'(x) = 3x^2 + 4x - 5",
            "topic": "calculus",
            "difficulty": "medium",
            "source": "jee_main_2025"
        },
        {
            "question": "Solve the quadratic equation 2x^2 + 5x - 3 = 0",
            "solution": "Using quadratic formula: x = (-5 ± √(25 + 24))/4 = (-5 ± 7)/4. So x = 1/2 or x = -3",
            "topic": "algebra",
            "difficulty": "easy",
            "source": "jee_main_2025"
        }
        # Add more problems...
    ]
    
    knowledge_base = MathKnowledgeBase()
    await knowledge_base.add_math_problems(sample_problems)
    return knowledge_base
```

### 3.5 DSPy-based Math Agent (app/agents/math_agent.py)

```python
import dspy
from typing import Dict, List, Optional
from app.services.vector_service import MathKnowledgeBase
from app.services.mcp_service import WebSearchMCP

class MathSolver(dspy.Signature):
    """Solve mathematical problems with step-by-step explanations"""
    question = dspy.InputField(desc="Mathematical problem to solve")
    context = dspy.InputField(desc="Relevant context from knowledge base")
    solution = dspy.OutputField(desc="Step-by-step solution")
    confidence = dspy.OutputField(desc="Confidence level (0-1)")

class StepByStepReasoning(dspy.Signature):
    """Break down complex problems into steps"""
    problem = dspy.InputField(desc="Complex mathematical problem")
    steps = dspy.OutputField(desc="List of solution steps")

class SolutionValidator(dspy.Signature):
    """Validate mathematical solutions"""
    problem = dspy.InputField(desc="Original problem")
    solution = dspy.OutputField(desc="Proposed solution")
    is_correct = dspy.OutputField(desc="True if solution is correct")
    feedback = dspy.OutputField(desc="Validation feedback")

class MathRoutingAgent:
    def __init__(self, knowledge_base: MathKnowledgeBase, web_search: WebSearchMCP):
        # Configure DSPy with Gemini
        lm = dspy.Google(model="gemini-1.5-pro", api_key=settings.GEMINI_API_KEY, max_tokens=2000)
        dspy.settings.configure(lm=lm)
        
        self.knowledge_base = knowledge_base
        self.web_search = web_search
        
        # Initialize DSPy modules
        self.solver = dspy.ChainOfThought(MathSolver)
        self.reasoner = dspy.ChainOfThought(StepByStepReasoning)
        self.validator = dspy.ChainOfThought(SolutionValidator)
        
        # Human feedback storage
        self.feedback_history = []
    
    async def solve_problem(self, question: str, user_id: str) -> Dict:
        """Main routing logic for mathematical problem solving"""
        
        # Step 1: Check knowledge base
        similar_problems = await self.knowledge_base.search_similar_problems(
            question, limit=3, score_threshold=0.8
        )
        
        if similar_problems:
            # Use knowledge base solution
            context = self._format_context(similar_problems)
            result = await self._solve_with_context(question, context)
            result["source"] = "knowledge_base"
            result["similar_problems"] = similar_problems
        else:
            # Fallback to web search
            search_results = await self.web_search.search_math_content(question)
            if search_results:
                context = self._format_search_context(search_results)
                result = await self._solve_with_context(question, context)
                result["source"] = "web_search"
                result["search_results"] = search_results
            else:
                # Generate solution without context
                result = await self._solve_without_context(question)
                result["source"] = "generated"
        
        # Validate solution
        validation = self.validator(
            problem=question,
            solution=result["solution"]
        )
        
        result["validation"] = {
            "is_correct": validation.is_correct,
            "feedback": validation.feedback
        }
        
        return result
    
    async def _solve_with_context(self, question: str, context: str) -> Dict:
        """Solve problem using retrieved context"""
        solution_result = self.solver(
            question=question,
            context=context
        )
        
        return {
            "solution": solution_result.solution,
            "confidence": float(solution_result.confidence),
            "reasoning_type": "context_based"
        }
    
    async def _solve_without_context(self, question: str) -> Dict:
        """Solve problem without external context"""
        # Break down into steps first
        steps_result = self.reasoner(problem=question)
        
        # Solve with step-by-step approach
        solution_result = self.solver(
            question=question,
            context=f"Problem breakdown: {steps_result.steps}"
        )
        
        return {
            "solution": solution_result.solution,
            "confidence": float(solution_result.confidence),
            "reasoning_type": "step_by_step",
            "steps": steps_result.steps
        }
    
    def _format_context(self, similar_problems: List[Dict]) -> str:
        """Format similar problems as context"""
        context_parts = ["Similar problems and solutions:"]
        
        for i, problem in enumerate(similar_problems, 1):
            context_parts.append(
                f"{i}. Q: {problem['question']}\n"
                f"   A: {problem['solution']}\n"
                f"   Topic: {problem['topic']}"
            )
        
        return "\n".join(context_parts)
    
    def _format_search_context(self, search_results: List[Dict]) -> str:
        """Format web search results as context"""
        context_parts = ["Relevant web search results:"]
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"{i}. {result['title']}\n"
                f"   {result['snippet']}"
            )
        
        return "\n".join(context_parts)
    
    async def process_feedback(
        self, 
        question: str, 
        solution: str, 
        feedback: Dict,
        user_id: str
    ):
        """Process human feedback for continuous improvement"""
        feedback_entry = {
            "question": question,
            "solution": solution,
            "feedback": feedback,
            "user_id": user_id,
            "timestamp": dspy.utils.get_timestamp()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # If feedback is negative, trigger re-solution
        if feedback.get("rating", 0) < 3:
            await self._improve_solution(feedback_entry)
    
    async def _improve_solution(self, feedback_entry: Dict):
        """Improve solution based on feedback"""
        # This would implement DSPy optimization based on feedback
        # For now, we'll add it to training data for future optimization
        pass
```

### 3.6 Model Context Protocol (MCP) Server (app/services/mcp_service.py)

```python
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
```

### 3.7 Human-in-the-Loop System (app/services/feedback_service.py)

```python
import asyncio
import uuid
from typing import Dict, List, Optional, Callable, Awaitable
from pydantic import BaseModel
from datetime import datetime

class FeedbackRequest(BaseModel):
    id: str
    question: str
    solution: str
    user_id: str
    timestamp: datetime
    requires_human_review: bool = False

class HumanFeedback(BaseModel):
    request_id: str
    rating: int  # 1-5 scale
    comments: str
    corrections: Optional[str] = None
    reviewer_id: str
    timestamp: datetime

class HumanInputRequest:
    def __init__(self, question: str, context: Dict = None):
        self.id = str(uuid.uuid4())
        self.question = question
        self.context = context or {}
        self._response_future = asyncio.Future()
    
    async def response(self) -> str:
        """Wait for human response"""
        return await self._response_future
    
    def set_response(self, response: str):
        """Provide human response"""
        self._response_future.set_result(response)

class HumanInLoopManager:
    def __init__(self):
        self.pending_requests: Dict[str, HumanInputRequest] = {}
        self.request_queue = asyncio.Queue()
        self.feedback_history: List[HumanFeedback] = []
        self.quality_threshold = 0.7
        
    async def evaluate_solution_quality(
        self, 
        question: str, 
        solution: str,
        confidence: float
    ) -> bool:
        """Determine if solution needs human review"""
        
        # Criteria for human review:
        # 1. Low confidence score
        # 2. Complex mathematical concepts
        # 3. Previous feedback indicates issues
        
        if confidence < self.quality_threshold:
            return True
        
        # Check for complex topics that usually need review
        complex_indicators = [
            "integral", "derivative", "limit", "proof",
            "theorem", "advanced", "calculus"
        ]
        
        if any(indicator in question.lower() for indicator in complex_indicators):
            return True
        
        return False
    
    async def request_human_feedback(
        self,
        question: str,
        solution: str,
        user_id: str,
        context: Dict = None
    ) -> HumanFeedback:
        """Request human feedback on a solution"""
        
        request = FeedbackRequest(
            id=str(uuid.uuid4()),
            question=question,
            solution=solution,
            user_id=user_id,
            timestamp=datetime.now(),
            requires_human_review=True
        )
        
        # Add to queue for human reviewers
        await self.request_queue.put({
            "type": "feedback_request",
            "request": request,
            "context": context
        })
        
        # Create human input request
        human_request = HumanInputRequest(
            question=f"Please review this mathematical solution:\n\n"
                    f"Question: {question}\n\n"
                    f"Solution: {solution}\n\n"
                    f"Please provide:\n"
                    f"1. Rating (1-5)\n"
                    f"2. Comments\n"
                    f"3. Corrections if needed",
            context={"request_id": request.id}
        )
        
        self.pending_requests[request.id] = human_request
        
        # Wait for human response (with timeout)
        try:
            response = await asyncio.wait_for(
                human_request.response(),
                timeout=300  # 5 minutes timeout
            )
            
            # Parse response and create feedback
            feedback = self._parse_human_response(response, request.id)
            self.feedback_history.append(feedback)
            
            return feedback
            
        except asyncio.TimeoutError:
            # Handle timeout - maybe use automated review
            return await self._automated_fallback_review(request)
    
    def _parse_human_response(
        self, 
        response: str, 
        request_id: str
    ) -> HumanFeedback:
        """Parse human feedback response"""
        # Simple parsing - in production, use structured input
        lines = response.strip().split('\n')
        
        rating = 3  # Default
        comments = response
        corrections = None
        
        # Try to extract structured data
        for line in lines:
            if line.startswith("Rating:"):
                try:
                    rating = int(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("Comments:"):
                comments = line.split(":", 1)[1].strip()
            elif line.startswith("Corrections:"):
                corrections = line.split(":", 1)[1].strip()
        
        return HumanFeedback(
            request_id=request_id,
            rating=rating,
            comments=comments,
            corrections=corrections,
            reviewer_id="human_reviewer",  # In production, track actual reviewer
            timestamp=datetime.now()
        )
    
    async def _automated_fallback_review(
        self, 
        request: FeedbackRequest
    ) -> HumanFeedback:
        """Automated review when human review times out"""
        return HumanFeedback(
            request_id=request.id,
            rating=3,  # Neutral rating
            comments="Automated review - human reviewer not available",
            reviewer_id="automated_system",
            timestamp=datetime.now()
        )
    
    async def get_pending_requests(self) -> List[Dict]:
        """Get pending feedback requests for human reviewers"""
        requests = []
        try:
            while True:
                request = self.request_queue.get_nowait()
                requests.append(request)
        except asyncio.QueueEmpty:
            pass
        
        return requests
    
    async def submit_feedback(
        self, 
        request_id: str, 
        feedback_data: Dict
    ):
        """Submit feedback from human reviewer"""
        if request_id in self.pending_requests:
            # Format feedback response
            response = (
                f"Rating: {feedback_data.get('rating', 3)}\n"
                f"Comments: {feedback_data.get('comments', '')}\n"
                f"Corrections: {feedback_data.get('corrections', '')}"
            )
            
            # Set the response
            self.pending_requests[request_id].set_response(response)
            
            # Clean up
            del self.pending_requests[request_id]
```

### 3.8 FastAPI Main Application (app/main.py)

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.guardrails import InputGuardrails, OutputGuardrails, GuardrailsConfig
from app.agents.math_agent import MathRoutingAgent
from app.services.vector_service import MathKnowledgeBase, load_jee_dataset
from app.services.mcp_service import WebSearchMCP
from app.services.feedback_service import HumanInLoopManager
from typing import Dict
import asyncio

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
knowledge_base = None
math_agent = None
guardrails_input = None
guardrails_output = None
human_loop_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global knowledge_base, math_agent, guardrails_input, guardrails_output, human_loop_manager
    
    # Initialize knowledge base
    knowledge_base = await load_jee_dataset()
    
    # Initialize web search
    web_search = WebSearchMCP(settings.TAVILY_API_KEY)
    
    # Initialize math agent
    math_agent = MathRoutingAgent(knowledge_base, web_search)
    
    # Initialize guardrails
    guardrails_config = GuardrailsConfig()
    guardrails_input = InputGuardrails(guardrails_config)
    guardrails_output = OutputGuardrails()
    
    # Initialize human-in-loop
    human_loop_manager = HumanInLoopManager()

@app.post("/api/v1/solve")
async def solve_math_problem(request: Dict):
    """Main endpoint for solving mathematical problems"""
    question = request.get("question", "")
    user_id = request.get("user_id", "anonymous")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        # Input guardrails
        await guardrails_input.validate_input(question, user_id)
        
        # Solve problem
        result = await math_agent.solve_problem(question, user_id)
        
        # Output guardrails
        output_validation = await guardrails_output.validate_output(
            result["solution"]
        )
        
        if output_validation["status"] == "flagged":
            raise HTTPException(
                status_code=400,
                detail=f"Output validation failed: {output_validation['reason']}"
            )
        
        # Check if human review needed
        needs_review = await human_loop_manager.evaluate_solution_quality(
            question, result["solution"], result["confidence"]
        )
        
        result["needs_human_review"] = needs_review
        
        # If needs review and enabled, request feedback
        if needs_review and request.get("enable_human_review", True):
            feedback_task = asyncio.create_task(
                human_loop_manager.request_human_feedback(
                    question, result["solution"], user_id
                )
            )
            result["feedback_request_id"] = "pending"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/feedback")
async def submit_feedback(request: Dict):
    """Submit feedback on a solution"""
    question = request.get("question")
    solution = request.get("solution")
    feedback = request.get("feedback")
    user_id = request.get("user_id", "anonymous")
    
    await math_agent.process_feedback(question, solution, feedback, user_id)
    
    return {"status": "feedback_received"}

@app.get("/api/v1/pending-reviews")
async def get_pending_reviews():
    """Get pending human reviews"""
    requests = await human_loop_manager.get_pending_requests()
    return {"pending_requests": requests}

@app.post("/api/v1/submit-review")
async def submit_review(request: Dict):
    """Submit human review"""
    request_id = request.get("request_id")
    feedback_data = request.get("feedback")
    
    await human_loop_manager.submit_feedback(request_id, feedback_data)
    
    return {"status": "review_submitted"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}
```

### 3.9 React Frontend Structure

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx
│   │   ├── FeedbackForm.tsx
│   │   ├── ReviewPanel.tsx
│   │   └── LoadingSpinner.tsx
│   ├── services/
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── hooks/
│   │   ├── useChat.ts
│   │   └── useFeedback.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── index.tsx
│   └── styles/
│       └── globals.css
├── package.json
└── vite.config.ts
```

### 3.10 React Components

**Chat Interface (src/components/ChatInterface.tsx):**

```typescript
import React, { useState } from 'react';
import { useMathSolver } from '../hooks/useChat';
import { MathProblem, SolutionResponse } from '../types';
import FeedbackForm from './FeedbackForm';
import LoadingSpinner from './LoadingSpinner';

const ChatInterface: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [solutions, setSolutions] = useState<SolutionResponse[]>([]);
  const { solveProblem, loading, error } = useMathSolver();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    try {
      const response = await solveProblem({
        question,
        user_id: 'user123', // In production, get from auth
        enable_human_review: true
      });
      
      setSolutions([...solutions, response]);
      setQuestion('');
    } catch (err) {
      console.error('Error solving problem:', err);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {solutions.map((solution, index) => (
          <div key={index} className="solution-card">
            <div className="question">
              <strong>Q:</strong> {solution.question}
            </div>
            <div className="solution">
              <strong>Solution:</strong>
              <div className="solution-content">
                {solution.solution}
              </div>
              <div className="meta-info">
                <span>Source: {solution.source}</span>
                <span>Confidence: {(solution.confidence * 100).toFixed(1)}%</span>
                {solution.needs_human_review && (
                  <span className="review-badge">Under Review</span>
                )}
              </div>
            </div>
            <FeedbackForm 
              question={solution.question}
              solution={solution.solution}
              onFeedback={(feedback) => {
                // Handle feedback submission
                console.log('Feedback submitted:', feedback);
              }}
            />
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <div className="input-group">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your mathematical problem..."
            className="question-input"
            disabled={loading}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? <LoadingSpinner /> : 'Solve'}
          </button>
        </div>
      </form>
      
      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}
    </div>
  );
};

export default ChatInterface;
```

### 3.11 JEE Benchmark Evaluation (app/utils/evaluation.py)

```python
import json
import asyncio
from typing import Dict, List, Tuple
from app.agents.math_agent import MathRoutingAgent
from datasets import load_dataset

class JEEBenchmarkEvaluator:
    def __init__(self, math_agent: MathRoutingAgent):
        self.math_agent = math_agent
        self.results = []
    
    async def run_evaluation(self, dataset_name: str = "PhysicsWallahAI/JEE-Main-2025-Math") -> Dict:
        """Run evaluation on JEE Main 2025 dataset"""
        
        # Load dataset (in production, integrate with HuggingFace datasets)
        dataset = await self._load_jee_dataset(dataset_name)
        
        total_problems = len(dataset)
        correct_answers = 0
        total_confidence = 0
        
        print(f"Evaluating on {total_problems} problems...")
        
        for i, problem in enumerate(dataset):
            print(f"Progress: {i+1}/{total_problems}")
            
            try:
                # Solve problem
                result = await self.math_agent.solve_problem(
                    problem["question"], 
                    "evaluation_user"
                )
                
                # Evaluate answer
                is_correct = self._evaluate_answer(
                    problem["correct_answer"], 
                    result["solution"]
                )
                
                if is_correct:
                    correct_answers += 1
                
                total_confidence += result["confidence"]
                
                # Store detailed result
                self.results.append({
                    "problem_id": i,
                    "question": problem["question"],
                    "correct_answer": problem["correct_answer"],
                    "agent_solution": result["solution"],
                    "is_correct": is_correct,
                    "confidence": result["confidence"],
                    "source": result["source"]
                })
                
            except Exception as e:
                print(f"Error evaluating problem {i}: {e}")
                self.results.append({
                    "problem_id": i,
                    "question": problem["question"],
                    "error": str(e),
                    "is_correct": False,
                    "confidence": 0
                })
        
        # Calculate metrics
        accuracy = correct_answers / total_problems
        avg_confidence = total_confidence / total_problems
        
        evaluation_report = {
            "total_problems": total_problems,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "results": self.results
        }
        
        # Save results
        await self._save_results(evaluation_report)
        
        return evaluation_report
    
    async def _load_jee_dataset(self, dataset_name: str) -> List[Dict]:
        """Load JEE dataset (simplified - in production use HuggingFace datasets)"""
        # This would load from the actual dataset
        # For now, return sample data
        return [
            {
                "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
                "correct_answer": "3x^2 + 4x - 5",
                "topic": "calculus",
                "difficulty": "medium"
            },
            {
                "question": "Solve 2x^2 + 5x - 3 = 0",
                "correct_answer": "x = 1/2, x = -3",
                "topic": "algebra", 
                "difficulty": "easy"
            }
        ]
    
    def _evaluate_answer(self, correct_answer: str, agent_solution: str) -> bool:
        """Evaluate if agent's solution matches correct answer"""
        # Simplified evaluation - in production, use more sophisticated matching
        # Could integrate with mathematical expression parsers
        
        # Normalize answers for comparison
        correct_normalized = self._normalize_answer(correct_answer)
        agent_normalized = self._normalize_answer(agent_solution)
        
        # Check for exact match or key terms
        return correct_normalized in agent_normalized or \
               any(term in agent_normalized for term in correct_normalized.split())
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        return answer.lower().replace(" ", "").replace("=", "")
    
    async def _save_results(self, report: Dict):
        """Save evaluation results"""
        with open("jee_benchmark_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nEvaluation Complete!")
        print(f"Accuracy: {report['accuracy']:.2%}")
        print(f"Average Confidence: {report['average_confidence']:.2f}")
        print(f"Results saved to jee_benchmark_results.json")
```

## 4. Deployment Configuration

### 4.1 Docker Compose (docker-compose.yml)

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=math_agent
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - postgres
      - qdrant
      - redis
    volumes:
      - ./backend:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm run dev

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=math_agent
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  qdrant_data:
```

## 5. Sample Questions for Testing

### Knowledge Base Questions (should find in vector DB):
1. "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1"
2. "Solve the quadratic equation 2x^2 + 5x - 3 = 0"
3. "Calculate the integral of sin(x)cos(x) dx"

### Web Search Questions (not in knowledge base):
1. "How do you solve a differential equation using Laplace transforms?"
2. "What is the application of complex analysis in electrical engineering?"
3. "Explain the proof of Fermat's Last Theorem"

## 6. Next Steps for Implementation

1. **Start with Backend Core**: Set up FastAPI, databases, and basic API endpoints
2. **Implement Guardrails**: Add input/output validation and safety measures
3. **Create Knowledge Base**: Load mathematical dataset into Qdrant
4. **Develop Math Agent**: Implement DSPy-based routing and solving logic
5. **Add MCP Server**: Integrate web search capabilities
6. **Build Frontend**: Create React interface for user interaction
7. **Implement Human Loop**: Add feedback system and review interface
8. **Add Evaluation**: Implement JEE benchmark testing
9. **Testing & Optimization**: Test all components and optimize performance
10. **Deployment**: Deploy using Docker containers

## 7. Additional Resources

- **DSPy Documentation**: https://dspy.ai
- **LangGraph Tutorials**: https://langchain-ai.github.io/langgraph/
- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **JEE Dataset**: https://huggingface.co/datasets/PhysicsWallahAI/JEE-Main-2025-Math
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/

This implementation guide provides a complete foundation for building the Mathematical Routing Agent as specified in your assignment requirements. Each component is modular and can be developed and tested independently.