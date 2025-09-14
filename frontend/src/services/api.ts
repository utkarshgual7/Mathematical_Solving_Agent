import { MathProblem, SolutionResponse, FeedbackData, ImageUploadResponse, ProcessingStatus } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export class MathAgentAPI {
  static async uploadImage(imageFile: File): Promise<ImageUploadResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/upload-image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload image: ${response.statusText}`);
    }

    return await response.json();
  }

  static async solveProblem(problem: MathProblem, userId: string = 'anonymous'): Promise<SolutionResponse> {
    const requestBody: any = {
      question: problem.question,
      user_id: userId,
      enable_human_review: true,
    };

    if (problem.imageUrl) {
      requestBody.image_url = problem.imageUrl;
    }

    const response = await fetch(`${API_BASE_URL}/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Failed to solve problem: ${response.statusText}`);
    }

    return await response.json();
  }

  static async getProcessingStatus(taskId: string): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get status: ${response.statusText}`);
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