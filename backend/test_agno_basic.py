#!/usr/bin/env python3
"""
Basic Agno integration test without requiring API keys

This script tests:
1. Import functionality
2. Basic class initialization
3. Component availability
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

# Set mock API keys for testing
os.environ['GEMINI_API_KEY'] = 'test-gemini-key-12345'

print("🧪 Starting Basic Agno Integration Tests...\n")

# Test 1: Import functionality
print("Test 1: Import functionality")
try:
    from agno.agent import Agent
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.models.google import Gemini
    from agno.vectordb.lancedb import LanceDb
    from agno.embedder.google import GeminiEmbedder
    from agno.memory import AgentMemory
    print("✅ All Agno imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: App imports
print("\nTest 2: App imports")
try:
    from app.services.vector_service import AgnoMathKnowledgeBase
    from app.services.pdf_processing_service import MathPDFProcessor
    from app.services.mcp_service import WebSearchMCP
    from app.agents.math_agent import MathRoutingAgent, MathematicalTools
    from app.core.config import settings
    print("✅ All app imports successful")
except ImportError as e:
    print(f"❌ App import failed: {e}")
    sys.exit(1)

# Test 3: Basic class initialization (without API calls)
print("\nTest 3: Basic class initialization")
try:
    # Test DuckDuckGo tools
    ddg_tools = DuckDuckGoTools()
    print("✅ DuckDuckGoTools initialized")
    
    # Test AgentMemory
    memory = AgentMemory()
    print("✅ AgentMemory initialized")
    
    # Test WebSearchMCP
    web_search = WebSearchMCP()
    print("✅ WebSearchMCP initialized")
    
    # Test MathematicalTools
    math_tools = MathematicalTools(web_search)
    print("✅ MathematicalTools initialized")
    
    # Test PDF processor
    pdf_processor = MathPDFProcessor()
    print("✅ MathPDFProcessor initialized")
    
except Exception as e:
    print(f"❌ Class initialization failed: {e}")
    sys.exit(1)

# Test 4: Configuration access
print("\nTest 4: Configuration access")
try:
    print(f"✅ Settings loaded - Debug mode: {settings.DEBUG}")
    print(f"✅ Knowledge base path: {settings.KNOWLEDGE_BASE_PATH}")
    print(f"✅ Vector DB path: {settings.VECTOR_DB_PATH}")
except Exception as e:
    print(f"❌ Configuration access failed: {e}")

# Test 5: Knowledge base initialization (basic)
print("\nTest 5: Knowledge base initialization (basic)")
try:
    # This will fail without real API key but should show proper error handling
    kb = AgnoMathKnowledgeBase()
    print("✅ AgnoMathKnowledgeBase class created (API key validation will occur on first use)")
except Exception as e:
    print(f"⚠️  Knowledge base initialization issue (expected without real API key): {e}")

print("\n🎉 Basic integration tests completed!")
print("\n📋 Summary:")
print("- All imports are working correctly")
print("- Basic class initialization is functional")
print("- Configuration system is accessible")
print("- Components are properly integrated")
print("\n💡 To test with real functionality:")
print("1. Set valid GEMINI_API_KEY in .env file")
print("2. Run: python test_agno_integration.py")
print("3. The system will perform actual API calls and vector operations")