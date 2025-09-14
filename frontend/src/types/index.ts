export interface MathProblem {
  question: string;
  topic?: string;
  difficulty?: string;
  image?: File;
  imageUrl?: string;
}

export interface SolutionResponse {
  question: string;
  solution: string;
  confidence: number;
  source: 'knowledge_base' | 'web_search' | 'generated';
  needs_human_review: boolean;
  similar_problems?: SimilarProblem[];
  search_results?: SearchResult[];
  validation: {
    is_correct: boolean;
    feedback: string;
  };
  steps?: string;
  imageUrl?: string;
  extractedText?: string;
  processingStatus?: ProcessingStatus;
}

export interface SimilarProblem {
  score: number;
  question: string;
  solution: string;
  topic: string;
  difficulty: string;
}

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  score: number;
}

export interface FeedbackData {
  rating: number;
  comments: string;
  corrections?: string;
}

export interface ProcessingStatus {
  stage: 'uploading' | 'extracting_text' | 'searching_knowledge_base' | 'searching_web' | 'generating_solution' | 'completed';
  message: string;
  progress?: number;
}

export interface ImageUploadResponse {
  imageUrl: string;
  extractedText: string;
  success: boolean;
  error?: string;
}