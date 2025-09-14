# Mathematical Routing Agent

A comprehensive AI system for solving mathematical problems with step-by-step explanations, built with FastAPI, DSPy, React, and Qdrant.

## Features

- **AI Gateway with Guardrails**: Input/output validation focused on mathematics-only queries
- **Vector Database Knowledge Base**: Using Qdrant with embeddings from sentence-transformers
- **DSPy-based Math Agent**: Step-by-step mathematical problem solving and solution validation
- **Model Context Protocol (MCP) Server**: Advanced mathematical web search using Tavily
- **Human-in-the-Loop Feedback System**: Async review requests, feedback parsing, and continuous learning
- **JEE Benchmark Evaluation**: Assessing system accuracy on JEE Main 2025 dataset
- **React Frontend**: Interactive chat interface with inline feedback forms

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Chat Interface │  │ Feedback UI │  │ Admin Panel │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                          HTTP/WebSocket
                              │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ API Gateway │  │ Auth System │  │ Guardrails  │         │
│  │ Routing     │  │             │  │ Layer       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Math Agent  │  │ Knowledge   │  │ Web Search  │         │
│  │ (DSPy)      │  │ Base RAG    │  │ MCP Server  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Human-Loop  │  │ Feedback    │  │ Evaluation  │         │
│  │ Manager     │  │ Processor   │  │ Engine      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Vector DB   │  │ PostgreSQL  │  │ Redis Cache │         │
│  │ (Qdrant)    │  │ (Metadata)  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Backend Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── math_agent.py
│   │   │   ├── feedback.py
│   │   │   └── knowledge.py
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── guardrails.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── math_agent.py
│   │   ├── knowledge_retriever.py
│   │   └── web_search.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_service.py
│   │   ├── feedback_service.py
│   │   └── mcp_service.py
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py
│       └── evaluation.py
├── requirements.txt
└── docker-compose.yml
```

## Frontend Structure

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx
│   │   ├── FeedbackForm.tsx
│   │   └── ReviewPanel.tsx
│   ├── services/
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── hooks/
│   │   ├── useChat.ts
│   │   └── useFeedback.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── styles/
│       └── globals.css
├── package.json
└── vite.config.ts
```

## Setup and Installation

### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=
   ```

### Frontend

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the Application

### Using Docker (Recommended)

```
docker-compose up --build
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup

1. Start the backend:
   ```
   cd backend
   uvicorn app.main:app --reload
   ```

2. Start the frontend:
   ```
   cd frontend
   npm run dev
   ```

## API Endpoints

- `POST /api/v1/solve` - Solve a mathematical problem
- `POST /api/v1/feedback` - Submit feedback on a solution
- `GET /api/v1/pending-reviews` - Get pending human reviews
- `POST /api/v1/submit-review` - Submit human review
- `GET /health` - Health check

## Development

### Backend Testing

Run the JEE benchmark evaluation:
```bash
python -m app.utils.evaluation
```

### Frontend Development

The React frontend uses Vite for fast development and hot reloading.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.