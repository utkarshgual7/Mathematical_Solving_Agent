# Knowledge Base Directory

This directory contains the mathematical knowledge base for the Agno-powered AI agent.

## Directory Structure

```
knowledge_base/
├── pdfs/           # Original PDF files containing mathematical Q&A
├── processed/      # Processed and extracted text from PDFs
├── embeddings/     # Vector embeddings and LanceDB storage
├── logs/          # Processing logs and metadata
└── README.md      # This file
```

## Usage

### Adding New PDFs
1. Place PDF files in the `pdfs/` directory
2. The system will automatically process them and extract mathematical content
3. Processed content will be stored in `processed/` as structured JSON
4. Vector embeddings will be generated and stored in `embeddings/`

### Supported PDF Content
- Mathematical problems and solutions
- Step-by-step explanations
- Formula derivations
- Conceptual explanations
- Practice problems with answers

### Processing Pipeline
1. **PDF Extraction**: Extract text and mathematical expressions from PDFs
2. **Content Parsing**: Identify questions, solutions, and explanations
3. **Embedding Generation**: Create vector embeddings for semantic search
4. **Knowledge Base Integration**: Store in LanceDB for fast retrieval

### File Naming Convention
- Original PDFs: `topic_level_source.pdf` (e.g., `algebra_basic_textbook.pdf`)
- Processed files: `topic_level_source_processed.json`
- Logs: `processing_YYYYMMDD_HHMMSS.log`

## Integration with Agno

The knowledge base integrates with Agno's PDFUrlKnowledgeBase and LanceDB for:
- Semantic search across mathematical content
- Context-aware problem solving
- Dynamic few-shot learning from similar problems
- Hybrid search combining vector similarity and keyword matching