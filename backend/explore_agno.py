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
    from agno.models.openai import OpenAIChat
    print("✓ agno.models.openai.OpenAIChat imported successfully")
except ImportError as e:
    print(f"✗ agno.models.openai.OpenAIChat failed: {e}")

try:
    from agno.vectordb.lancedb import LanceDb
    print("✓ agno.vectordb.lancedb.LanceDb imported successfully")
except ImportError as e:
    print(f"✗ agno.vectordb.lancedb.LanceDb failed: {e}")

try:
    from agno.embedder.openai import OpenAIEmbedder
    print("✓ agno.embedder.openai.OpenAIEmbedder imported successfully")
except ImportError as e:
    print(f"✗ agno.embedder.openai.OpenAIEmbedder failed: {e}")