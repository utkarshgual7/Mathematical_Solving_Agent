import React from 'react';
import ChatInterface from './components/ChatInterface';
import './styles/globals.css';

const App: React.FC = () => {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Mathematical Routing Agent</h1>
        <p>Solve complex mathematical problems with AI assistance</p>
      </header>
      <main className="app-main">
        <ChatInterface />
      </main>
    </div>
  );
};

export default App;