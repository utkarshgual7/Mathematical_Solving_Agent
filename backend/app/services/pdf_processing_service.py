import os
import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

import fitz  # PyMuPDF
import sympy as sp
from sympy.parsing.latex import parse_latex
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.google import GeminiEmbedder

from app.core.config import settings

class MathPDFProcessor:
    """Service for processing mathematical PDF documents and extracting Q&A content"""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        self.pdfs_path = self.kb_path / "pdfs"
        self.processed_path = self.kb_path / "processed"
        self.embeddings_path = self.kb_path / "embeddings"
        self.logs_path = self.kb_path / "logs"
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Mathematical patterns for content extraction
        self.question_patterns = [
            r'(?i)(?:question|problem|q\.?|\d+\.?)\s*[:\-]?\s*(.+?)(?=(?:answer|solution|sol\.?|a\.?)|$)',
            r'(?i)find\s+(.+?)(?=(?:answer|solution)|$)',
            r'(?i)solve\s+(.+?)(?=(?:answer|solution)|$)',
            r'(?i)calculate\s+(.+?)(?=(?:answer|solution)|$)',
            r'(?i)determine\s+(.+?)(?=(?:answer|solution)|$)'
        ]
        
        self.answer_patterns = [
            r'(?i)(?:answer|solution|sol\.?|a\.?)\s*[:\-]?\s*(.+?)(?=(?:question|problem|q\.?|\d+\.?)|$)',
            r'(?i)(?:therefore|thus|hence)\s*[,:]?\s*(.+?)(?=(?:question|problem)|$)'
        ]
        
        # Mathematical expression patterns
        self.math_patterns = [
            r'\$([^$]+)\$',  # LaTeX inline math
            r'\\\[([^\]]+)\\\]',  # LaTeX display math
            r'\\begin\{equation\}(.+?)\\end\{equation\}',  # LaTeX equations
            r'\\begin\{align\}(.+?)\\end\{align\}',  # LaTeX align
            r'([a-zA-Z]\s*[=]\s*[^\n]+)',  # Simple equations
        ]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for PDF processing"""
        logger = logging.getLogger('pdf_processor')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = self.logs_path / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                
                # Also extract text from images if any
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    # This would require OCR for mathematical expressions in images
                    # For now, we'll focus on text-based content
                    pass
            
            doc.close()
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def parse_mathematical_content(self, text: str) -> List[Dict]:
        """Parse mathematical questions and answers from text"""
        problems = []
        
        # Split text into sections (by page breaks, chapter markers, etc.)
        sections = self._split_into_sections(text)
        
        for section_idx, section in enumerate(sections):
            section_problems = self._extract_qa_pairs(section, section_idx)
            problems.extend(section_problems)
        
        return problems
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""
        # Split by common section markers
        section_markers = [
            r'\n\s*(?:Chapter|Section|Part)\s+\d+',
            r'\n\s*\d+\.\d+',  # Numbered sections
            r'\n\s*Problem\s+\d+',
            r'\n\s*Exercise\s+\d+',
            r'\f'  # Form feed (page break)
        ]
        
        sections = [text]  # Start with full text
        
        for pattern in section_markers:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section, flags=re.IGNORECASE)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections
        
        return sections
    
    def _extract_qa_pairs(self, text: str, section_idx: int) -> List[Dict]:
        """Extract question-answer pairs from a text section"""
        problems = []
        
        # Method 1: Pattern-based extraction
        qa_pairs = self._pattern_based_extraction(text)
        
        # Method 2: Heuristic-based extraction for mathematical content
        if not qa_pairs:
            qa_pairs = self._heuristic_extraction(text)
        
        for idx, (question, answer) in enumerate(qa_pairs):
            if question and answer:
                problem = {
                    "id": f"section_{section_idx}_problem_{idx}",
                    "question": self._clean_text(question),
                    "answer": self._clean_text(answer),
                    "solution_steps": self._extract_solution_steps(answer),
                    "mathematical_expressions": self._extract_math_expressions(question + " " + answer),
                    "topic": self._infer_topic(question),
                    "difficulty": self._estimate_difficulty(question, answer),
                    "source_section": section_idx,
                    "metadata": {
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_method": "pattern_based" if qa_pairs else "heuristic"
                    }
                }
                problems.append(problem)
        
        return problems
    
    def _pattern_based_extraction(self, text: str) -> List[Tuple[str, str]]:
        """Extract Q&A pairs using regex patterns"""
        qa_pairs = []
        
        # Try to find question-answer patterns
        for q_pattern in self.question_patterns:
            questions = re.findall(q_pattern, text, re.DOTALL | re.IGNORECASE)
            
            for a_pattern in self.answer_patterns:
                answers = re.findall(a_pattern, text, re.DOTALL | re.IGNORECASE)
                
                # Pair questions with answers
                min_len = min(len(questions), len(answers))
                for i in range(min_len):
                    qa_pairs.append((questions[i], answers[i]))
        
        return qa_pairs
    
    def _heuristic_extraction(self, text: str) -> List[Tuple[str, str]]:
        """Extract Q&A pairs using heuristic methods"""
        qa_pairs = []
        
        # Split by numbered items
        items = re.split(r'\n\s*\d+\.', text)
        
        for item in items[1:]:  # Skip first empty split
            # Look for question indicators
            if any(indicator in item.lower() for indicator in ['find', 'solve', 'calculate', 'determine', 'prove']):
                # Split by solution indicators
                parts = re.split(r'(?i)(?:solution|answer|sol\.)', item, maxsplit=1)
                if len(parts) == 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                    qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def _extract_solution_steps(self, solution_text: str) -> List[str]:
        """Extract step-by-step solution from answer text"""
        steps = []
        
        # Look for numbered steps
        step_patterns = [
            r'(?:Step|step)\s+(\d+)[:\.]\s*([^\n]+)',
            r'(\d+)[\.):]\s*([^\n]+)',
            r'(?:First|Second|Third|Finally)[,:]\s*([^\n]+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                steps.extend([match[1] if isinstance(match, tuple) else match for match in matches])
                break
        
        # If no numbered steps, split by sentences
        if not steps:
            sentences = re.split(r'[.!?]+', solution_text)
            steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return steps[:10]  # Limit to 10 steps
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text"""
        expressions = []
        
        for pattern in self.math_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            expressions.extend(matches)
        
        # Clean and validate expressions
        cleaned_expressions = []
        for expr in expressions:
            cleaned = self._clean_math_expression(expr)
            if cleaned and self._is_valid_math_expression(cleaned):
                cleaned_expressions.append(cleaned)
        
        return list(set(cleaned_expressions))  # Remove duplicates
    
    def _clean_math_expression(self, expr: str) -> str:
        """Clean mathematical expression"""
        # Remove extra whitespace and common LaTeX commands
        cleaned = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', expr)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _is_valid_math_expression(self, expr: str) -> bool:
        """Check if expression is a valid mathematical expression"""
        try:
            # Try to parse with sympy
            sp.sympify(expr)
            return True
        except:
            # Check for basic mathematical content
            math_indicators = ['=', '+', '-', '*', '/', '^', 'x', 'y', 'z', '∫', '∑', '√']
            return any(indicator in expr for indicator in math_indicators) and len(expr) > 2
    
    def _infer_topic(self, question: str) -> str:
        """Infer mathematical topic from question text"""
        topic_keywords = {
            'algebra': ['equation', 'solve', 'variable', 'polynomial', 'linear', 'quadratic'],
            'calculus': ['derivative', 'integral', 'limit', 'differentiate', 'integrate'],
            'geometry': ['triangle', 'circle', 'angle', 'area', 'volume', 'perimeter'],
            'trigonometry': ['sin', 'cos', 'tan', 'trigonometric', 'angle'],
            'statistics': ['probability', 'mean', 'median', 'standard deviation', 'distribution'],
            'number_theory': ['prime', 'factor', 'divisible', 'gcd', 'lcm']
        }
        
        question_lower = question.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
        
        return 'general'
    
    def _estimate_difficulty(self, question: str, answer: str) -> str:
        """Estimate difficulty level based on content complexity"""
        complexity_indicators = {
            'basic': ['add', 'subtract', 'multiply', 'divide', 'simple'],
            'intermediate': ['solve', 'find', 'calculate', 'determine'],
            'advanced': ['prove', 'derive', 'analyze', 'complex', 'advanced']
        }
        
        text = (question + " " + answer).lower()
        
        # Count mathematical expressions
        math_count = len(self._extract_math_expressions(text))
        
        # Check for advanced keywords
        for level, keywords in reversed(complexity_indicators.items()):
            if any(keyword in text for keyword in keywords):
                if level == 'advanced' or math_count > 3:
                    return 'advanced'
                elif level == 'intermediate' or math_count > 1:
                    return 'intermediate'
                else:
                    return 'basic'
        
        return 'basic'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        return text.strip()
    
    def process_pdf_file(self, pdf_path: str) -> Dict:
        """Process a single PDF file and extract mathematical content"""
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                raise ValueError("No text extracted from PDF")
            
            # Parse mathematical content
            problems = self.parse_mathematical_content(text)
            
            # Create processed file
            pdf_name = Path(pdf_path).stem
            processed_file = self.processed_path / f"{pdf_name}_processed.json"
            
            # Ensure processed directory exists
            self.processed_path.mkdir(parents=True, exist_ok=True)
            
            # Save processed content
            processed_data = {
                "source_file": pdf_path,
                "processed_at": datetime.now().isoformat(),
                "total_problems": len(problems),
                "problems": problems,
                "metadata": {
                    "text_length": len(text),
                    "processing_version": "1.0"
                }
            }
            
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Successfully processed {len(problems)} problems from {pdf_path}")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def create_agno_knowledge_base(self, processed_files: List[str] = None) -> PDFReader:
        """Create Agno knowledge base from processed mathematical content"""
        try:
            # Ensure embeddings directory exists
            self.embeddings_path.mkdir(parents=True, exist_ok=True)
            
            # Create LanceDB vector store
            vector_db = LanceDb(
                uri=str(self.embeddings_path / "math_knowledge.lancedb"),
                table_name="mathematical_problems",
                search_type=SearchType.hybrid,
                embedder=GeminiEmbedder(
                    id="gemini-embedding-001",
                    api_key=settings.GEMINI_API_KEY
                )
            )
            
            # If no specific files provided, process all in processed directory
            if processed_files is None:
                processed_files = list(self.processed_path.glob("*_processed.json"))
            
            # Load processed content into knowledge base
            all_problems = []
            for file_path in processed_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_problems.extend(data['problems'])
            
            # Create knowledge base content for Agno
            knowledge_content = self._format_for_agno(all_problems)
            
            # Insert documents into LanceDB
            vector_db.insert(documents=knowledge_content)
            
            self.logger.info(f"Created Agno knowledge base with {len(all_problems)} problems")
            
            return PDFReader(
                path=str(self.pdfs_path),
                vector_db=vector_db
            )
            
        except Exception as e:
            self.logger.error(f"Error creating Agno knowledge base: {str(e)}")
            raise
    
    def _format_for_agno(self, problems: List[Dict]) -> str:
        """Format mathematical problems for Agno knowledge base"""
        formatted_content = []
        
        for problem in problems:
            content = f"""Problem ID: {problem['id']}
Topic: {problem['topic']}
Difficulty: {problem['difficulty']}

Question: {problem['question']}

Answer: {problem['answer']}

Solution Steps:
"""
            
            for i, step in enumerate(problem.get('solution_steps', []), 1):
                content += f"{i}. {step}\n"
            
            if problem.get('mathematical_expressions'):
                content += "\nMathematical Expressions:\n"
                for expr in problem['mathematical_expressions']:
                    content += f"- {expr}\n"
            
            content += "\n" + "="*50 + "\n\n"
            formatted_content.append(content)
        
        return "\n".join(formatted_content)
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDF files in the pdfs directory"""
        pdf_files = list(self.pdfs_path.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning("No PDF files found in pdfs directory")
            return []
        
        processed_data = []
        for pdf_file in pdf_files:
            try:
                data = self.process_pdf_file(str(pdf_file))
                processed_data.append(data)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {str(e)}")
                continue
        
        return processed_data


# Utility functions for integration
def get_pdf_processor() -> MathPDFProcessor:
    """Get configured PDF processor instance"""
    return MathPDFProcessor()

async def process_new_pdfs() -> PDFReader:
    """Process any new PDFs and return updated knowledge base"""
    processor = get_pdf_processor()
    processor.process_all_pdfs()
    return processor.create_agno_knowledge_base()