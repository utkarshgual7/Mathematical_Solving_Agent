import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json

class MathKnowledgeBase:
    def __init__(self, host: str = "localhost", port: int = 6333):
        # Use in-memory Qdrant client instead of connecting to external server
        self.client = QdrantClient(":memory:")
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