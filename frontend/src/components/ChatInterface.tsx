import React, { useState, useRef, useCallback } from 'react';
import { useMathSolver } from '../hooks/useChat';
import { MathProblem, SolutionResponse, ProcessingStatus } from '../types';
import { MathAgentAPI } from '../services/api';
import FeedbackForm from './FeedbackForm';
import ProcessingLoader from './ProcessingLoader';

const ChatInterface: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [solutions, setSolutions] = useState<SolutionResponse[]>([]);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const { solveProblem, loading, error } = useMathSolver();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    setUploadedImage(file);
    setImagePreview(URL.createObjectURL(file));
    
    try {
      setProcessingStatus({
        stage: 'uploading',
        message: 'Uploading image...'
      });

      const uploadResponse = await MathAgentAPI.uploadImage(file);
      
      if (uploadResponse.success) {
        setProcessingStatus({
          stage: 'extracting_text',
          message: 'Text extracted successfully'
        });
        
        setQuestion(uploadResponse.extractedText || 'Please solve this mathematical problem from the image.');
        
        setTimeout(() => {
          setProcessingStatus(null);
        }, 1000);
      }
    } catch (err) {
      console.error('Error uploading image:', err);
      setProcessingStatus(null);
      alert('Failed to upload image. Please try again.');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleImageUpload(files[0]);
    }
  }, [handleImageUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleImageUpload(files[0]);
    }
  }, [handleImageUpload]);

  const removeImage = useCallback(() => {
    setUploadedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    try {
      setProcessingStatus({
        stage: 'searching_knowledge_base',
        message: 'Searching knowledge base...'
      });

      const problemData: MathProblem = {
        question,
        imageUrl: imagePreview || undefined
      };

      const response = await solveProblem({
        ...problemData,
        user_id: 'user123', // In production, get from auth
        enable_human_review: true
      });
      
      // Simulate processing stages based on response source
      if (response.source === 'knowledge_base') {
        setProcessingStatus({
          stage: 'searching_knowledge_base',
          message: 'Found similar problems in knowledge base'
        });
      } else if (response.source === 'web_search') {
        setProcessingStatus({
          stage: 'searching_web',
          message: 'Searching web for solutions...'
        });
      } else {
        setProcessingStatus({
          stage: 'generating_solution',
          message: 'Generating solution...'
        });
      }

      setTimeout(() => {
        setProcessingStatus({
          stage: 'completed',
          message: 'Solution ready!'
        });
        
        setTimeout(() => {
          setProcessingStatus(null);
          setSolutions([...solutions, { ...response, imageUrl: imagePreview || undefined }]);
          setQuestion('');
          removeImage();
        }, 1000);
      }, 1500);
      
    } catch (err) {
      console.error('Error solving problem:', err);
      setProcessingStatus(null);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {solutions.map((solution, index) => (
          <div key={index} className="solution-card">
            <div className="question">
              <strong>Q:</strong> {solution.question}
              {solution.imageUrl && (
                <div className="question-image">
                  <img src={solution.imageUrl} alt="Problem" className="image-preview" />
                </div>
              )}
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
      
      <ProcessingLoader 
        status={processingStatus!} 
        isVisible={!!processingStatus} 
      />
      
      <form onSubmit={handleSubmit} className="input-form">
        {/* Image Upload Area */}
        <div 
          className={`image-upload-area ${isDragOver ? 'drag-over' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          {imagePreview ? (
            <div className="image-container">
              <img src={imagePreview} alt="Preview" className="image-preview" />
              <button 
                type="button" 
                className="remove-image" 
                onClick={(e) => {
                  e.stopPropagation();
                  removeImage();
                }}
              >
                ×
              </button>
            </div>
          ) : (
            <>
              <div className="upload-icon">📷</div>
              <div className="upload-text">
                Drag and drop an image here, or click to select
                <br />
                <small>Supports mathematical diagrams, equations, and handwritten problems</small>
              </div>
            </>
          )}
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        <div className="input-group">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your mathematical problem or upload an image..."
            className="question-input"
            disabled={loading || !!processingStatus}
          />
          <button type="submit" disabled={loading || !question.trim() || !!processingStatus}>
            {loading || processingStatus ? 'Processing...' : 'Solve'}
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