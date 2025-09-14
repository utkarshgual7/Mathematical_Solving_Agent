import pytest
from app.agents.math_agent import MathRoutingAgent
from app.services.vector_service import MathKnowledgeBase
from app.services.mcp_service import WebSearchMCP
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_knowledge_base():
    kb = Mock(spec=MathKnowledgeBase)
    kb.search_similar_problems = AsyncMock(return_value=[])
    return kb

@pytest.fixture
def mock_web_search():
    ws = Mock(spec=WebSearchMCP)
    ws.search_math_content = AsyncMock(return_value=[])
    return ws

@pytest.fixture
def math_agent(mock_knowledge_base, mock_web_search):
    return MathRoutingAgent(mock_knowledge_base, mock_web_search)

@pytest.mark.asyncio
async def test_solve_simple_problem(math_agent):
    # Mock the DSPy modules
    math_agent.solver = Mock()
    math_agent.solver.return_value = Mock(
        solution="The solution is x = 5",
        confidence="0.9"
    )
    
    math_agent.validator = Mock()
    math_agent.validator.return_value = Mock(
        is_correct="True",
        feedback="Solution is correct"
    )
    
    # Test solving a simple problem
    result = await math_agent.solve_problem("What is 2 + 2?", "test_user")
    
    # Assertions
    assert "solution" in result
    assert "confidence" in result
    assert "validation" in result
    assert result["source"] in ["knowledge_base", "web_search", "generated"]
    
    # Verify mocks were called
    assert math_agent.solver.called
    assert math_agent.validator.called

if __name__ == "__main__":
    pytest.main([__file__])