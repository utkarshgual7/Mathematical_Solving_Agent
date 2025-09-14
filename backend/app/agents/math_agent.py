import dspy
from typing import Dict, List, Optional
from app.services.vector_service import MathKnowledgeBase
from app.services.mcp_service import WebSearchMCP
from app.core.config import settings

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
        lm = dspy.Google(model="gemini-2.5-flash", api_key=settings.GEMINI_API_KEY)
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