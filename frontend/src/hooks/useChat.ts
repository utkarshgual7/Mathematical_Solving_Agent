import { useState } from 'react';
import { MathAgentAPI } from '../services/api';
import { MathProblem, SolutionResponse } from '../types';

export const useMathSolver = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const solveProblem = async (problem: MathProblem & { user_id?: string; enable_human_review?: boolean }): Promise<SolutionResponse> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await MathAgentAPI.solveProblem(
        { question: problem.question, topic: problem.topic, difficulty: problem.difficulty },
        problem.user_id
      );
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { solveProblem, loading, error };
};