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