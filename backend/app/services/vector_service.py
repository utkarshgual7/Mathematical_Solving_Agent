import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import asyncio
from datetime import datetime

from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.document import Document

from app.core.config import settings
from app.services.pdf_processing_service import MathPDFProcessor

class AgnoMathKnowledgeBase:
    """Agno-based mathematical knowledge base using LanceDB"""
    
    def __init__(self, embeddings_path: str = "knowledge_base/embeddings"):
        self.embeddings_path = Path(embeddings_path)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Agno components
        self.embedder = OpenAIEmbedder(
            id="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
        
        self.vector_db = LanceDb(
            uri=str(self.embeddings_path / "math_knowledge.lancedb"),
            table_name="mathematical_problems",
            search_type=SearchType.hybrid,
            embedder=self.embedder
        )
        
        self.pdf_processor = MathPDFProcessor()
        self.knowledge_base = None
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize Agno knowledge base from processed PDFs"""
        try:
            # Check for processed PDFs
            processed_path = Path("knowledge_base/processed")
            pdf_urls = []
            
            if processed_path.exists():
                processed_files = list(processed_path.glob("*_processed.json"))
                
                # Convert processed JSON files to documents for Agno
                documents = []
                for file_path in processed_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        for problem in data.get('problems', []):
                            # Create document for each mathematical problem
                            doc_content = self._format_problem_for_agno(problem)
                            
                            document = Document(
                                content=doc_content,
                                meta={
                                    "question": problem.get('question', ''),
                                    "solution": problem.get('solution', ''),
                                    "topic": problem.get('topic', 'general'),
                                    "difficulty": problem.get('difficulty', 'medium'),
                                    "source": data.get('metadata', {}).get('filename', 'unknown'),
                                    "problem_type": problem.get('type', 'mathematical'),
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                            documents.append(document)
                    
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
                
                # Add documents to vector database
                if documents:
                    self.vector_db.insert(documents)
                    print(f"Added {len(documents)} mathematical problems to Agno knowledge base")
            
            # Create PDFUrlKnowledgeBase for compatibility
            self.knowledge_base = PDFUrlKnowledgeBase(
                urls=pdf_urls,
                vector_db=self.vector_db
            )
            
        except Exception as e:
            print(f"Error initializing knowledge base: {e}")
            self.knowledge_base = None
    
    async def add_math_problems(self, problems: List[Dict]):
        """Add mathematical problems to Agno knowledge base"""
        try:
            documents = []
            
            for problem in problems:
                # Format problem for Agno
                doc_content = self._format_problem_for_agno(problem)
                
                # Create Agno document
                document = Document(
                    content=doc_content,
                    meta={
                        "question": problem.get("question", ""),
                        "solution": problem.get("solution", ""),
                        "topic": problem.get("topic", "general"),
                        "difficulty": problem.get("difficulty", "medium"),
                        "source": problem.get("source", "manual_input"),
                        "problem_type": "mathematical",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                documents.append(document)
            
            # Insert into vector database
            if documents:
                self.vector_db.insert(documents)
                print(f"Added {len(documents)} problems to Agno knowledge base")
                
                # Refresh knowledge base
                self._initialize_knowledge_base()
                
        except Exception as e:
            print(f"Error adding problems to knowledge base: {e}")
    
    async def search_similar_problems(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Search for similar problems using Agno knowledge base"""
        try:
            if not self.knowledge_base:
                return []
            
            # Search using Agno knowledge base
            search_results = self.knowledge_base.search(
                query=query,
                limit=limit
            )
            
            # Format results for compatibility
            results = []
            for result in search_results:
                if hasattr(result, 'meta') and result.meta:
                    results.append({
                        "score": getattr(result, 'score', 0.8),  # Default score if not available
                        "question": result.meta.get("question", ""),
                        "solution": result.meta.get("solution", ""),
                        "topic": result.meta.get("topic", "general"),
                        "difficulty": result.meta.get("difficulty", "medium"),
                        "source": result.meta.get("source", "unknown"),
                        "content": result.content if hasattr(result, 'content') else ""
                    })
                elif isinstance(result, dict):
                    # Handle dict results
                    results.append({
                        "score": result.get("score", 0.8),
                        "question": result.get("question", ""),
                        "solution": result.get("solution", ""),
                        "topic": result.get("topic", "general"),
                        "difficulty": result.get("difficulty", "medium"),
                        "source": result.get("source", "unknown"),
                        "content": result.get("content", "")
                    })
            
            # Filter by score threshold
            filtered_results = [
                result for result in results 
                if result.get("score", 0) >= score_threshold
            ]
            
            return filtered_results[:limit]
            
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    def _format_problem_for_agno(self, problem: Dict) -> str:
        """Format mathematical problem for Agno document storage"""
        components = [
            f"Question: {problem.get('question', '')}",
            f"Solution: {problem.get('solution', '')}",
            f"Topic: {problem.get('topic', 'general')}",
            f"Difficulty: {problem.get('difficulty', 'medium')}"
        ]
        
        # Add additional context if available
        if problem.get('steps'):
            components.append(f"Solution Steps: {problem['steps']}")
        
        if problem.get('concepts'):
            components.append(f"Key Concepts: {', '.join(problem['concepts'])}")
        
        if problem.get('formulas'):
            components.append(f"Formulas Used: {', '.join(problem['formulas'])}")
        
        return "\n".join(filter(None, components))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Agno knowledge base"""
        try:
            stats = {
                "framework": "Agno with LanceDB",
                "embedder": "OpenAI text-embedding-3-small",
                "vector_db_type": "LanceDB",
                "knowledge_base_active": bool(self.knowledge_base)
            }
            
            # Count processed PDFs and problems
            processed_path = Path("knowledge_base/processed")
            if processed_path.exists():
                processed_files = list(processed_path.glob("*_processed.json"))
                stats["processed_pdfs"] = len(processed_files)
                
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
            
            return stats
            
        except Exception as e:
            return {
                "error": f"Error getting stats: {str(e)}",
                "framework": "Agno with LanceDB"
            }
    
    async def refresh_from_pdfs(self) -> Dict[str, Any]:
        """Refresh knowledge base from processed PDFs"""
        try:
            # Re-initialize knowledge base
            self._initialize_knowledge_base()
            
            stats = self.get_stats()
            
            return {
                "status": "success",
                "message": "Knowledge base refreshed from processed PDFs",
                "stats": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error refreshing knowledge base: {str(e)}"
            }

# Sample dataset loading for Agno
async def load_sample_math_dataset():
    """Load sample mathematical dataset into Agno knowledge base"""
    sample_problems = [
        {
            "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "solution": "f'(x) = 3x^2 + 4x - 5",
            "topic": "calculus",
            "difficulty": "medium",
            "source": "sample_dataset",
            "concepts": ["derivatives", "polynomial functions"],
            "formulas": ["power rule"]
        },
        {
            "question": "Solve the quadratic equation 2x^2 + 5x - 3 = 0",
            "solution": "Using quadratic formula: x = (-5 ± √(25 + 24))/4 = (-5 ± 7)/4. So x = 1/2 or x = -3",
            "topic": "algebra",
            "difficulty": "easy",
            "source": "sample_dataset",
            "concepts": ["quadratic equations", "quadratic formula"],
            "formulas": ["quadratic formula: x = (-b ± √(b²-4ac))/2a"]
        },
        {
            "question": "Find the integral of ∫(3x^2 + 2x - 1)dx",
            "solution": "∫(3x^2 + 2x - 1)dx = x^3 + x^2 - x + C",
            "topic": "calculus",
            "difficulty": "medium",
            "source": "sample_dataset",
            "concepts": ["integration", "polynomial integration"],
            "formulas": ["power rule for integration"]
        }
    ]
    
    knowledge_base = AgnoMathKnowledgeBase()
    await knowledge_base.add_math_problems(sample_problems)
    return knowledge_base

# Backward compatibility alias
MathKnowledgeBase = AgnoMathKnowledgeBase