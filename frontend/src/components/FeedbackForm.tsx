import React, { useState } from 'react';
import { FeedbackData } from '../types';
import { MathAgentAPI } from '../services/api';

interface FeedbackFormProps {
  question: string;
  solution: string;
  onFeedback: (feedback: FeedbackData) => void;
}

const FeedbackForm: React.FC<FeedbackFormProps> = ({ question, solution, onFeedback }) => {
  const [rating, setRating] = useState(0);
  const [comments, setComments] = useState('');
  const [corrections, setCorrections] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const feedback: FeedbackData = {
      rating,
      comments,
      corrections: corrections || undefined
    };

    try {
      await MathAgentAPI.submitFeedback(question, solution, feedback);
      onFeedback(feedback);
      setSubmitted(true);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    }
  };

  if (submitted) {
    return <div className="feedback-submitted">Thank you for your feedback!</div>;
  }

  return (
    <div className="feedback-form">
      <h4>Provide Feedback</h4>
      <form onSubmit={handleSubmit}>
        <div className="rating">
          <label>Rating: </label>
          {[1, 2, 3, 4, 5].map((star) => (
            <span 
              key={star}
              className={`star ${star <= rating ? 'filled' : ''}`}
              onClick={() => setRating(star)}
            >
              ★
            </span>
          ))}
        </div>
        
        <div className="form-group">
          <label htmlFor="comments">Comments:</label>
          <textarea
            id="comments"
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            placeholder="Please provide your feedback..."
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="corrections">Corrections (if any):</label>
          <textarea
            id="corrections"
            value={corrections}
            onChange={(e) => setCorrections(e.target.value)}
            placeholder="If you see any errors, please provide the correct solution..."
          />
        </div>
        
        <button type="submit">Submit Feedback</button>
      </form>
    </div>
  );
};

export default FeedbackForm;