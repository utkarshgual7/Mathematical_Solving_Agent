import { MathProblem, SolutionResponse, FeedbackData } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export class MathAgentAPI {
  static async solveProblem(problem: MathProblem, userId: string = 'anonymous'): Promise<SolutionResponse> {
    const response = await fetch(`${API_BASE_URL}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: problem.question,
        user_id: userId,
        enable_human_review: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to solve problem: ${response.statusText}`);
    }

    return await response.json();
  }

  static async submitFeedback(
    question: string,
    solution: string,
    feedback: FeedbackData,
    userId: string = 'anonymous'
  ): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        solution,
        feedback,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to submit feedback: ${response.statusText}`);
    }
  }
}