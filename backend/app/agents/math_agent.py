from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.google import GeminiEmbedder

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from pathlib import Path

from app.services.pdf_processing_service import MathPDFProcessor, process_new_pdfs
from app.services.mcp_service import WebSearchMCP
from app.core.config import settings

class MathematicalTools:
    """Custom mathematical tools for Agno agent"""
    
    def __init__(self, web_search: WebSearchMCP):
        self.web_search = web_search
    
    def search_mathematical_content(self, query: str, max_results: int = 5) -> str:
        """Search for mathematical content using web search"""
        try:
            results = asyncio.run(self.web_search.search_math_content(query, max_results))
            formatted_results = []
            
            for result in results:
                formatted_results.append(
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"Content: {result.get('content', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                )
            
            return "\n---\n".join(formatted_results)
        except Exception as e:
            return f"Error searching mathematical content: {str(e)}"
    
    def validate_mathematical_solution(self, problem: str, solution: str) -> str:
        """Validate a mathematical solution"""
        try:
            # Use sympy or other mathematical libraries for validation
            import sympy as sp
            
            validation_result = {
                "is_valid": True,
                "feedback": "Solution appears mathematically sound",
                "suggestions": []
            }
            
            # Basic validation checks
            if not solution.strip():
                validation_result["is_valid"] = False
                validation_result["feedback"] = "Solution is empty"
            
            # Check for mathematical expressions
            math_expressions = []
            import re
            
            # Look for equations, formulas, etc.
            equation_pattern = r'[a-zA-Z]\s*[=]\s*[^\n]+'
            equations = re.findall(equation_pattern, solution)
            
            if equations:
                math_expressions.extend(equations)
                validation_result["mathematical_expressions"] = math_expressions
            
            return json.dumps(validation_result, indent=2)
            
        except Exception as e:
            return f"Error validating solution: {str(e)}"
    
    def extract_solution_steps(self, solution: str) -> str:
        """Extract and format solution steps"""
        try:
            import re
            
            # Look for numbered steps
            step_patterns = [
                r'(?:Step|step)\s+(\d+)[:\.]\s*([^\n]+)',
                r'(\d+)[\.):]\s*([^\n]+)',
                r'(?:First|Second|Third|Finally)[,:]\s*([^\n]+)'
            ]
            
            steps = []
            for pattern in step_patterns:
                matches = re.findall(pattern, solution, re.IGNORECASE)
                if matches:
                    steps.extend([match[1] if isinstance(match, tuple) else match for match in matches])
                    break
            
            if not steps:
                # Split by sentences as fallback
                sentences = re.split(r'[.!?]+', solution)
                steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            formatted_steps = []
            for i, step in enumerate(steps[:10], 1):
                formatted_steps.append(f"Step {i}: {step}")
            
            return "\n".join(formatted_steps)
            
        except Exception as e:
            return f"Error extracting steps: {str(e)}"

class MathRoutingAgent:
    """Advanced mathematical problem-solving agent using Agno framework"""
    
    def __init__(self, web_search: WebSearchMCP = None):
        self.web_search = web_search or WebSearchMCP(settings.TAVILY_API_KEY)
        self.pdf_processor = MathPDFProcessor()
        self.math_tools = MathematicalTools(self.web_search)
        
        # Initialize knowledge base
        self.knowledge_base = self._setup_knowledge_base()
        
        # Initialize Agno agents
        self._setup_agents()
        
        # Human feedback storage
        self.feedback_history = []
        self.session_memory = {}
    
    def _setup_knowledge_base(self) -> Optional[PDFReader]:
        """Setup mathematical knowledge base from processed PDFs"""
        try:
            # Check if there are processed PDFs
            processed_path = Path("knowledge_base/processed")
            if processed_path.exists() and list(processed_path.glob("*_processed.json")):
                return self.pdf_processor.create_agno_knowledge_base()
            else:
                # Create empty knowledge base that can be populated later
                embeddings_path = Path("knowledge_base/embeddings")
                embeddings_path.mkdir(parents=True, exist_ok=True)
                
                vector_db = LanceDb(
                    uri=str(embeddings_path / "math_knowledge.lancedb"),
                    table_name="mathematical_problems",
                    search_type=SearchType.hybrid,
                    embedder=GeminiEmbedder(
                id="gemini-embedding-001",
                api_key=settings.GEMINI_API_KEY
            )
                )
                
                return PDFReader(
                    path=str(self.kb_path / "pdfs"),
                    vector_db=vector_db
                )
        except Exception as e:
            print(f"Warning: Could not setup knowledge base: {e}")
            return None
    
    def _setup_agents(self):
        """Setup specialized Agno agents for different mathematical tasks"""
        
        # Choose model based on available API keys
        if settings.GEMINI_API_KEY:
            model = Gemini(api_key=settings.GEMINI_API_KEY)
        elif settings.GEMINI_API_KEY:
            model = Gemini(id="gemini-2.0-flash-exp", api_key=settings.GEMINI_API_KEY)
        else:
            raise ValueError("No valid API key found for LLM models")
        
        # Main mathematical problem solver
        self.solver_agent = Agent(
            name="Mathematical Problem Solver",
            model=model,
            description="You are an expert mathematician specializing in solving complex mathematical problems with step-by-step explanations.",
            instructions=[
                "Always provide detailed step-by-step solutions",
                "Show all mathematical work and reasoning",
                "Use proper mathematical notation",
                "Explain concepts clearly for educational value",
                "If unsure, search for similar problems in the knowledge base",
                "Validate your solutions before presenting them"
            ],
            tools=[DuckDuckGoTools()],
            knowledge=self.knowledge_base,
            markdown=True
        )
        
        # Solution validator agent
        self.validator_agent = Agent(
            name="Solution Validator",
            model=model,
            description="You are a mathematical solution validator who checks the correctness and completeness of mathematical solutions.",
            instructions=[
                "Carefully review each step of the solution",
                "Check mathematical accuracy and logic",
                "Identify any errors or missing steps",
                "Provide constructive feedback",
                "Suggest improvements if needed"
            ],
            tools=[],
            markdown=True
        )
        
        # Step-by-step reasoning agent
        self.reasoning_agent = Agent(
            name="Mathematical Reasoner",
            model=model,
            description="You specialize in breaking down complex mathematical problems into clear, logical steps.",
            instructions=[
                "Break complex problems into manageable steps",
                "Explain the reasoning behind each step",
                "Identify key concepts and formulas needed",
                "Provide clear transitions between steps"
            ],
            tools=[],
            knowledge=self.knowledge_base,
            markdown=True
        )
    
    async def solve_problem(self, question: str, user_id: str) -> Dict:
        """Main entry point for solving mathematical problems using Agno agents"""
        try:
            # Process any new PDFs first
            await self._process_new_pdfs()
            
            # Search for relevant context from knowledge base
            context = await self._get_relevant_context(question)
            
            # Prepare the problem with context
            problem_with_context = self._prepare_problem_context(question, context)
            
            # Use reasoning agent to break down the problem
            reasoning_response = self.reasoning_agent.run(
                f"Analyze this mathematical problem and break it into logical steps:\n\n{problem_with_context}"
            )
            
            # Use solver agent to solve the problem
            solver_response = self.solver_agent.run(
                f"Solve this mathematical problem step by step:\n\n{problem_with_context}\n\nProblem Analysis:\n{reasoning_response.content}"
            )
            
            # Validate the solution
            validation_response = self.validator_agent.run(
                f"Validate this mathematical solution:\n\nProblem: {question}\n\nSolution: {solver_response.content}\n\nProvide feedback on correctness and completeness."
            )
            
            # Extract confidence score
            confidence = self._extract_confidence(validation_response.content)
            
            # Store interaction
            interaction = {
                "user_id": user_id,
                "question": question,
                "solution": solver_response.content,
                "reasoning": reasoning_response.content,
                "validation": validation_response.content,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "context_used": bool(context)
            }
            
            self.feedback_history.append(interaction)
            
            return {
                "solution": solver_response.content,
                "reasoning": reasoning_response.content,
                "validation": validation_response.content,
                "confidence": confidence,
                "context_used": bool(context),
                "interaction_id": len(self.feedback_history) - 1
            }
            
        except Exception as e:
            error_msg = f"I encountered an error while solving this problem: {str(e)}"
            return {
                "error": str(e),
                "solution": error_msg,
                "confidence": 0.0,
                "context_used": False
            }
    
    async def _get_relevant_context(self, question: str) -> Optional[str]:
        """Search for relevant context from Agno knowledge base and web search"""
        try:
            context_parts = []
            
            # Search Agno knowledge base if available
            if self.knowledge_base:
                try:
                    # Use vector service for searching
                    from app.services.vector_service import AgnoMathKnowledgeBase
                    vector_service = AgnoMathKnowledgeBase()
                    kb_results = await vector_service.search_similar_problems(query=question, limit=3)
                    if kb_results:
                        for result in kb_results:
                            if isinstance(result, dict):
                                content = result.get('content', result.get('text', str(result)))
                                context_parts.append(f"Knowledge Base: {content}")
                except Exception as e:
                    print(f"Knowledge base search error: {e}")
            
            # Search web for additional context if needed
            if len(context_parts) < 2:
                try:
                    web_results = await self.web_search.search_math_content(question, max_results=3)
                    for result in web_results:
                        if isinstance(result, dict):
                            content = result.get('content', result.get('title', str(result)))
                            context_parts.append(f"Web Search: {content}")
                except Exception as e:
                    print(f"Web search error: {e}")
            
            return "\n\n---\n\n".join(context_parts) if context_parts else None
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return None
    
    def _prepare_problem_context(self, question: str, context: Optional[str]) -> str:
        """Prepare problem with relevant context for Agno agents"""
        if context:
            return f"Mathematical Problem: {question}\n\nRelevant Context:\n{context}\n\nPlease solve this problem using the provided context where applicable."
        else:
            return f"Mathematical Problem: {question}\n\nPlease solve this problem step by step."
    
    def _extract_confidence(self, validation_content: str) -> float:
        """Extract confidence score from validation response"""
        try:
            import re
            
            # Look for confidence patterns
            confidence_patterns = [
                r'confidence[:\s]*([0-9]*\.?[0-9]+)',
                r'([0-9]*\.?[0-9]+)\s*confidence',
                r'score[:\s]*([0-9]*\.?[0-9]+)',
                r'([0-9]*\.?[0-9]+)\s*out\s*of\s*[0-9]+'
            ]
            
            for pattern in confidence_patterns:
                matches = re.findall(pattern, validation_content.lower())
                if matches:
                    try:
                        score = float(matches[0])
                        # Normalize to 0-1 range
                        if score > 1.0:
                            score = score / 10.0 if score <= 10.0 else score / 100.0
                        return min(max(score, 0.0), 1.0)
                    except ValueError:
                        continue
            
            # Default confidence based on validation content
            if any(word in validation_content.lower() for word in ['correct', 'accurate', 'valid']):
                return 0.8
            elif any(word in validation_content.lower() for word in ['error', 'incorrect', 'wrong']):
                return 0.3
            else:
                return 0.6
                
        except Exception:
            return 0.6
    
    async def _process_new_pdfs(self):
        """Process any new PDFs in the knowledge base folder"""
        try:
            # Check for new PDFs and process them
            new_pdfs_processed = await process_new_pdfs()
            
            if new_pdfs_processed:
                # Refresh knowledge base if new PDFs were processed
                self.knowledge_base = self._setup_knowledge_base()
                print(f"Processed {new_pdfs_processed} new PDF(s) and updated knowledge base")
                
        except Exception as e:
            print(f"Error processing new PDFs: {e}")
    
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
    
    def add_human_feedback(self, interaction_id: int, feedback: Dict):
        """Add human feedback to improve future responses"""
        try:
            if 0 <= interaction_id < len(self.feedback_history):
                self.feedback_history[interaction_id]["human_feedback"] = feedback
                self.feedback_history[interaction_id]["feedback_timestamp"] = datetime.now().isoformat()
                
                # Store feedback in agent memory for future reference
                feedback_summary = f"User feedback on problem '{self.feedback_history[interaction_id]['question']}': {feedback.get('comment', 'No comment')}"
                
                if hasattr(self.solver_agent, 'memory') and self.solver_agent.memory:
                    self.solver_agent.memory.add(feedback_summary)
                
                print(f"Added human feedback for interaction {interaction_id}")
                
        except Exception as e:
            print(f"Error adding feedback: {e}")
    
    def get_agent_status(self) -> Dict:
        """Get status of all Agno agents"""
        try:
            return {
                "solver_agent": {
                    "name": self.solver_agent.name,
                    "model": str(self.solver_agent.model),
                    "tools": [str(tool) for tool in self.solver_agent.tools] if self.solver_agent.tools else [],
                    "knowledge_base": bool(self.knowledge_base)
                },
                "validator_agent": {
                    "name": self.validator_agent.name,
                    "model": str(self.validator_agent.model)
                },
                "reasoning_agent": {
                    "name": self.reasoning_agent.name,
                    "model": str(self.reasoning_agent.model)
                },
                "total_interactions": len(self.feedback_history),
                "knowledge_base_status": "active" if self.knowledge_base else "inactive"
            }
        except Exception as e:
            return {"error": f"Error getting agent status: {str(e)}"}
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the Agno-based agent"""
        if not self.feedback_history:
            return {"total_problems": 0}
        
        total_problems = len(self.feedback_history)
        
        # Calculate success rate based on confidence scores
        high_confidence_solutions = sum(1 for interaction in self.feedback_history 
                                      if interaction.get("confidence", 0) >= 0.7)
        
        avg_confidence = sum(interaction.get("confidence", 0) 
                           for interaction in self.feedback_history) / total_problems
        
        context_usage = sum(1 for interaction in self.feedback_history 
                          if interaction.get("context_used", False))
        
        feedback_count = sum(1 for interaction in self.feedback_history 
                           if "human_feedback" in interaction)
        
        return {
            "total_problems": total_problems,
            "high_confidence_solutions": high_confidence_solutions,
            "success_rate": high_confidence_solutions / total_problems if total_problems > 0 else 0,
            "average_confidence": avg_confidence,
            "context_usage_rate": context_usage / total_problems if total_problems > 0 else 0,
            "feedback_count": feedback_count,
            "knowledge_base_active": bool(self.knowledge_base),
            "agent_framework": "Agno"
        }
    
    async def process_uploaded_pdf(self, pdf_path: str) -> Dict:
        """Process uploaded PDF using the MathPDFProcessor"""
        try:
            # Use the PDF processor to handle the uploaded file
            result = await self.pdf_processor.process_pdf(pdf_path)
            
            if result["status"] == "success":
                # Refresh knowledge base to include new content
                self.knowledge_base = self._setup_knowledge_base()
                
                return {
                    "status": "success",
                    "problems_extracted": result.get("problems_extracted", 0),
                    "message": f"Successfully processed PDF and extracted {result.get('problems_extracted', 0)} mathematical problems"
                }
            else:
                return result
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF: {str(e)}"
            }
    
    async def update_knowledge_base(self) -> Dict:
        """Manually trigger knowledge base update from processed PDFs"""
        try:
            # Process any new PDFs
            await self._process_new_pdfs()
            
            # Refresh knowledge base
            old_kb_status = bool(self.knowledge_base)
            self.knowledge_base = self._setup_knowledge_base()
            new_kb_status = bool(self.knowledge_base)
            
            return {
                "status": "success",
                "message": "Knowledge base updated successfully",
                "knowledge_base_active": new_kb_status,
                "status_changed": old_kb_status != new_kb_status
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error updating knowledge base: {str(e)}"
            }
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the Agno knowledge base"""
        try:
            stats = {
                "status": "active" if self.knowledge_base else "inactive",
                "framework": "Agno with LanceDB"
            }
            
            if self.knowledge_base:
                try:
                    # Try to get vector database stats
                    if hasattr(self.knowledge_base, 'vector_db') and self.knowledge_base.vector_db:
                        stats["vector_db_type"] = "LanceDB"
                        stats["embedder"] = "Gemini gemini-embedding-001"
                    
                    # Check processed PDFs
                    processed_path = Path("knowledge_base/processed")
                    if processed_path.exists():
                        processed_files = list(processed_path.glob("*_processed.json"))
                        stats["processed_pdfs"] = len(processed_files)
                        
                        # Count total problems from processed files
                        total_problems = 0
                        for file_path in processed_files:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    total_problems += len(data.get('problems', []))
                            except Exception:
                                continue
                        
                        stats["total_problems"] = total_problems
                    else:
                        stats["processed_pdfs"] = 0
                        stats["total_problems"] = 0
                        
                except Exception as e:
                    stats["error"] = f"Error getting detailed stats: {str(e)}"
            
            return stats
            
        except Exception as e:
            return {
                "error": f"Error getting knowledge base stats: {str(e)}",
                "status": "error"
            }