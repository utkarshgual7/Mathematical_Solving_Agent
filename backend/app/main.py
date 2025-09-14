from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.guardrails import InputGuardrails, OutputGuardrails, GuardrailsConfig
from app.agents.math_agent import MathRoutingAgent
from app.services.vector_service import MathKnowledgeBase, load_jee_dataset
from app.services.mcp_service import WebSearchMCP
from app.services.feedback_service import HumanInLoopManager
from app.services.image_service import image_service
from typing import Dict
import asyncio
import os

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploaded images
if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Global services
knowledge_base = None
math_agent = None
guardrails_input = None
guardrails_output = None
human_loop_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global knowledge_base, math_agent, guardrails_input, guardrails_output, human_loop_manager
    
    try:
        print("Starting application initialization...")
        
        # Initialize knowledge base (this may take time to download models)
        print("Initializing knowledge base...")
        knowledge_base = await load_jee_dataset()
        print("Knowledge base initialized successfully")
        
        # Initialize web search
        print("Initializing web search...")
        web_search = WebSearchMCP(settings.TAVILY_API_KEY)
        
        # Initialize math agent
        print("Initializing math agent...")
        math_agent = MathRoutingAgent(web_search)
        
        # Initialize guardrails
        print("Initializing guardrails...")
        guardrails_config = GuardrailsConfig()
        guardrails_input = InputGuardrails(guardrails_config)
        guardrails_output = OutputGuardrails()
        
        # Initialize human-in-loop
        print("Initializing human-in-loop manager...")
        human_loop_manager = HumanInLoopManager()
        
        print("Application initialization completed successfully!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        # Continue startup even if some components fail
        print("Continuing with partial initialization...")

@app.post("/api/v1/upload-image")
async def upload_image(image: UploadFile = File(...)):
    """Upload and process image to extract mathematical content"""
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process the image
        result = await image_service.process_image(image)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to process image"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/v1/solve")
async def solve_math_problem(request: Dict):
    """Main endpoint for solving mathematical problems"""
    question = request.get("question", "")
    user_id = request.get("user_id", "anonymous")
    image_url = request.get("image_url", "")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Check if services are initialized
    if math_agent is None:
        raise HTTPException(status_code=503, detail="Math agent not initialized")
    if guardrails_input is None:
        raise HTTPException(status_code=503, detail="Guardrails not initialized")
    
    try:
        print(f"Solving problem: {question}")
        if image_url:
            print(f"With image: {image_url}")
        
        # Input guardrails
        await guardrails_input.validate_input(question, user_id)
        
        # Solve problem
        result = await math_agent.solve_problem(question, user_id)
        
        # Add image URL to result if provided
        if image_url:
            result["imageUrl"] = image_url
        
        # Output guardrails
        output_validation = await guardrails_output.validate_output(
            result["solution"]
        )
        
        if output_validation["status"] == "flagged":
            raise HTTPException(
                status_code=400,
                detail=f"Output validation failed: {output_validation['reason']}"
            )
        
        # Check if human review needed
        needs_review = await human_loop_manager.evaluate_solution_quality(
            question, result["solution"], result["confidence"]
        )
        
        result["needs_human_review"] = needs_review
        
        # If needs review and enabled, request feedback
        if needs_review and request.get("enable_human_review", True):
            feedback_task = asyncio.create_task(
                human_loop_manager.request_human_feedback(
                    question, result["solution"], user_id
                )
            )
            result["feedback_request_id"] = "pending"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/feedback")
async def submit_feedback(request: Dict):
    """Submit feedback on a solution"""
    question = request.get("question")
    solution = request.get("solution")
    feedback = request.get("feedback")
    user_id = request.get("user_id", "anonymous")
    
    await math_agent.process_feedback(question, solution, feedback, user_id)
    
    return {"status": "feedback_received"}

@app.get("/api/v1/pending-reviews")
async def get_pending_reviews():
    """Get pending human reviews"""
    requests = await human_loop_manager.get_pending_requests()
    return {"pending_requests": requests}

@app.post("/api/v1/submit-review")
async def submit_review(request: Dict):
    """Submit human review"""
    request_id = request.get("request_id")
    feedback_data = request.get("feedback")
    
    await human_loop_manager.submit_feedback(request_id, feedback_data)
    
    return {"status": "review_submitted"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}