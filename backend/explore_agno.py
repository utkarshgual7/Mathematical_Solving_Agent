import pkgutil
import agno

print("Agno package structure:")
for importer, modname, ispkg in pkgutil.walk_packages(agno.__path__, agno.__name__ + '.'):
    print(f"  {modname} {'(package)' if ispkg else '(module)'}")

print("\nTrying to import common modules:")
try:
    from agno.agent import Agent
    print("✓ agno.agent.Agent imported successfully")
except ImportError as e:
    print(f"✗ agno.agent.Agent failed: {e}")

try:
    from agno.tools.duckduckgo import DuckDuckGoTools
    print("✓ agno.tools.duckduckgo.DuckDuckGoTools imported successfully")
except ImportError as e:
    print(f"✗ agno.tools.duckduckgo.DuckDuckGoTools failed: {e}")

try:
    from agno.models.google import Gemini
    print("✓ agno.models.google.Gemini imported successfully")
except ImportError as e:
    print(f"✗ agno.models.google.Gemini failed: {e}")

try:
    from agno.vectordb.lancedb import LanceDb
    print("✓ agno.vectordb.lancedb.LanceDb imported successfully")
except ImportError as e:
    print(f"✗ agno.vectordb.lancedb.LanceDb failed: {e}")

try:
    from agno.embedder.google import GeminiEmbedder
    print("✓ agno.embedder.google.GeminiEmbedder imported successfully")
except ImportError as e:
    print(f"✗ agno.embedder.google.GeminiEmbedder failed: {e}")