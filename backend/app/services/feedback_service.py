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