import React from 'react';
import { ProcessingStatus } from '../types';

interface ProcessingLoaderProps {
  status: ProcessingStatus;
  isVisible: boolean;
}

const ProcessingLoader: React.FC<ProcessingLoaderProps> = ({ status, isVisible }) => {
  if (!isVisible) return null;

  const getStageIcon = (stage: ProcessingStatus['stage']) => {
    switch (stage) {
      case 'uploading':
        return '📤';
      case 'extracting_text':
        return '🔍';
      case 'searching_knowledge_base':
        return '📚';
      case 'searching_web':
        return '🌐';
      case 'generating_solution':
        return '🧠';
      case 'completed':
        return '✅';
      default:
        return '⏳';
    }
  };

  const getStageLabel = (stage: ProcessingStatus['stage']) => {
    switch (stage) {
      case 'uploading':
        return 'Uploading image...';
      case 'extracting_text':
        return 'Extracting text from image...';
      case 'searching_knowledge_base':
        return 'Searching knowledge base...';
      case 'searching_web':
        return 'Searching web for solutions...';
      case 'generating_solution':
        return 'Generating solution...';
      case 'completed':
        return 'Completed!';
      default:
        return 'Processing...';
    }
  };

  return (
    <div className="processing-loader">
      <div className="loader-content">
        <div className="loader-icon">
          {getStageIcon(status.stage)}
        </div>
        <div className="loader-text">
          <div className="stage-label">{getStageLabel(status.stage)}</div>
          <div className="status-message">{status.message}</div>
        </div>
        {status.progress !== undefined && (
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${status.progress}%` }}
            />
          </div>
        )}
      </div>
      <div className="loader-spinner">
        <div className="spinner"></div>
      </div>
    </div>
  );
};

export default ProcessingLoader;