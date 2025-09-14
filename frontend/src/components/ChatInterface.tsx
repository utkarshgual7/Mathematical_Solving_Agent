import React, { useState } from 'react';
import { useMathSolver } from '../hooks/useChat';
import { MathProblem, SolutionResponse } from '../types';
import FeedbackForm from './FeedbackForm';

const ChatInterface: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [solutions, setSolutions] = useState<SolutionResponse[]>([]);
  const { solveProblem, loading, error } = useMathSolver();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    try {
      const response = await solveProblem({
        question,
        user_id: 'user123', // In production, get from auth
        enable_human_review: true
      });
      
      setSolutions([...solutions, response]);
      setQuestion('');
    } catch (err) {
      console.error('Error solving problem:', err);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {solutions.map((solution, index) => (
          <div key={index} className="solution-card">
            <div className="question">
              <strong>Q:</strong> {solution.question}
            </div>
            <div className="solution">
              <strong>Solution:</strong>
              <div className="solution-content">
                {solution.solution}
              </div>
              <div className="meta-info">
                <span>Source: {solution.source}</span>
                <span>Confidence: {(solution.confidence * 100).toFixed(1)}%</span>
                {solution.needs_human_review && (
                  <span className="review-badge">Under Review</span>
                )}
              </div>
            </div>
            <FeedbackForm 
              question={solution.question}
              solution={solution.solution}
              onFeedback={(feedback) => {
                // Handle feedback submission
                console.log('Feedback submitted:', feedback);
              }}
            />
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <div className="input-group">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your mathematical problem..."
            className="question-input"
            disabled={loading}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? 'Solving...' : 'Solve'}
          </button>
        </div>
      </form>
      
      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}
    </div>
  );
};

export default ChatInterface;