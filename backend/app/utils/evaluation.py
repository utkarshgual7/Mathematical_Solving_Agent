import json
import asyncio
from typing import Dict, List, Tuple
from app.agents.math_agent import MathRoutingAgent

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
                "question": "Solve the quadratic equation 2x^2 + 5x - 3 = 0",
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