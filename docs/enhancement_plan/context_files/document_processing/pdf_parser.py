"""
PDF document parsing module for GraphRAG tutor.
"""
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
import pypdf
from pypdf import PdfReader
import pypdf.errors

from ..utils.text_processing import detect_language, clean_text
from .content_filter import ContentFilter
from .formula_extractor import FormulaExtractor
from .table_extractor import TableExtractor
from .diagram_analyzer import DiagramAnalyzer

logger = logging.getLogger(__name__)

# Add this custom exception at the module level (below imports)
class PDFReadError(Exception):
    """Custom exception for PDF reading errors."""
    pass

class PDFParser:
    """
    Class for parsing PDF documents and extracting text and metadata.
    """
    def __init__(self, 
                exclude_pages: Optional[List[int]] = None,
                llm_service=None,
                api_key: Optional[str] = None,
                filter_non_rag_content: bool = True):
        """
        Initialize the PDF parser.
        
        Args:
            exclude_pages: List of page numbers to exclude (0-indexed)
            llm_service: Optional LLM service for non-text element processing
            api_key: API key for external services (e.g., Gemini for diagrams)
            filter_non_rag_content: Whether to filter out non-RAG useful content
        """
        self.exclude_pages = exclude_pages or []
        self.llm_service = llm_service
        self.api_key = api_key
        self.filter_non_rag_content = filter_non_rag_content
        
        # Initialize component classes
        self.content_filter = ContentFilter()
        self.formula_extractor = FormulaExtractor(llm_service=llm_service)
        self.table_extractor = TableExtractor()
        self.diagram_analyzer = DiagramAnalyzer(llm_service=llm_service, api_key=api_key)
        
    def validate_pdf(self, file_path: str) -> bool:
        """
        Validate if the file is a valid PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if file is a valid PDF, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        if not file_path.lower().endswith('.pdf'):
            logger.error(f"File is not a PDF: {file_path}")
            return False
            
        try:
            with open(file_path, 'rb') as f:
                # Try to read the PDF file
                PdfReader(f)
            return True
        except pypdf.errors.PdfReadError as e:
            logger.error(f"Invalid or corrupted PDF file: {file_path}. Error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error validating PDF file: {file_path}. Error: {str(e)}")
            return False
    
    def extract_text(self, file_path: str) -> Tuple[str, Dict]:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple[str, Dict]: Extracted text and document metadata
        """
        # Add at the beginning of the method:
        logger.debug(f"Processing PDF: {file_path}")
        if not self.validate_pdf(file_path):
            raise ValueError(f"Invalid or corrupted PDF file: {file_path}")

        extracted_text = ""
        metadata = {}
        pages_text = []
        non_text_elements = {
            'tables': [],
            'diagrams': [],
            'formulas': []
        }

        # Add at the beginning of the method:
        logger.debug(f"Processing PDF: {file_path}")

        try:
            # Modify the 'with open...' block:
            with open(file_path, 'rb') as f:
                try:
                    pdf = PdfReader(f)
                except pypdf.errors.PdfReadError as e:
                    logger.error(f"Failed to read PDF {file_path}: {e}")
                    raise PDFReadError(f"Corrupted or unreadable PDF: {file_path}") from e

                # Extract metadata
                metadata = self._extract_metadata(pdf)
                
                # Modify the page iteration loop (inside the 'with open...' block):
                # First pass: Extract raw text from each page
                for i, page in enumerate(pdf.pages):
                    # Inside the loop: for i, page in enumerate(pdf.pages):
                    if i in self.exclude_pages:
                        pages_text.append("") # Keep placeholder for excluded pages
                        continue

                    page_text = "" # Initialize page_text for this iteration
                    try:
                        # Attempt to extract text from the current page
                        extracted = page.extract_text()
                        if extracted: # Check if text was actually extracted
                            page_text = extracted
                        else:
                            # Log if no text is extracted, but don't treat as error
                            logger.debug(f"No text extracted from page {i} in {file_path} (possibly image-based or empty).")
                            # page_text remains ""

                    except Exception as e:
                        # Log the error and ensure page_text remains ""
                        logger.warning(f"Could not extract text from page {i} in {file_path}: {e}")
                        # page_text remains ""

                    # Always append the result (either extracted text or "")
                    pages_text.append(page_text)

                # Add after the page loop (before language detection):
                num_pages = len(pdf.pages)
                logger.info(f"Extracted text from {num_pages} pages in {file_path}")

                # Detect language from a sample of text
                sample_text = "".join(pages_text[:min(5, len(pages_text))])
                language = detect_language(sample_text)
                metadata['language'] = language
                
                # Filter out TOC, index, and other non-RAG useful content
                if self.filter_non_rag_content:
                    filtered_pages, excluded_pages = self.content_filter.filter_document_pages(pages_text)
                    # Modify the existing logging for filtered pages (inside the 'if self.filter_non_rag_content:' block):
                    if excluded_pages: # Only log if pages were actually excluded
                        logger.info(f"Filtered out {len(excluded_pages)} non-RAG useful pages from {file_path}")
                    self.exclude_pages.extend(excluded_pages)
                    pages_text = filtered_pages
                    metadata['excluded_pages'] = self.exclude_pages
                
                # Extract and process tables
                temp_dir = os.path.join(os.path.dirname(file_path), 'temp_extracted')
                tables = [] # Initialize tables outside try block
                try:
                    os.makedirs(temp_dir, exist_ok=True)

                    # Extract tables within a try block
                    try:
                        tables = self.table_extractor.extract_tables_from_pdf(
                            file_path,
                            pages=[i for i, text in enumerate(pages_text) if text]
                        )
                        # Add this log after table extraction
                        logger.info(f"Extracted {len(tables)} tables from {file_path}")
                        if not tables: # Log if extraction succeeded but returned nothing
                             logger.warning(f"Table extraction produced no results for {file_path}")
                    except Exception as e:
                        logger.warning(f"Table extraction failed for {file_path}: {e}")
                        tables = [] # Ensure tables is empty list on failure

                    if tables:
                        # logger.info(f"Extracted {len(tables)} tables from PDF") # Removed duplicate log line
                        non_text_elements['tables'] = tables

                    # Extract diagrams
                    diagrams = self.diagram_analyzer.extract_diagrams_from_pdf(
                        file_path,
                        output_dir=temp_dir
                    )
                    # Add this log after diagram extraction
                    logger.info(f"Extracted {len(diagrams)} diagrams from {file_path}")
                    if diagrams:
                        # logger.info(f"Extracted {len(diagrams)} diagrams from PDF") # Removed duplicate log line
                        # Generate descriptions for diagrams
                        for i, diagram in enumerate(diagrams):
                            if diagram.get('file_path'):
                                description = self.diagram_analyzer.generate_diagram_description(
                                    diagram_data=diagram
                                )
                                diagram['description'] = description
                        non_text_elements['diagrams'] = diagrams
                        
                except Exception as e:
                    logger.error(f"Error extracting non-text elements: {str(e)}")
                
                # Extract formulas from text
                all_formulas = []
                for i, page_text in enumerate(pages_text):
                    if not page_text:
                        continue
                        
                    formulas = self.formula_extractor.extract_formulas(page_text)
                    for formula in formulas:
                        formula['page'] = i
                        all_formulas.append(formula)

                # Add this log after formula extraction loop
                logger.info(f"Extracted {len(all_formulas)} formulas from {file_path}")
                if all_formulas:
                    # logger.info(f"Extracted {len(all_formulas)} formulas from PDF") # Removed duplicate log line
                    non_text_elements['formulas'] = all_formulas

                # Join filtered pages into final text
                extracted_text = "\n\n".join(page for page in pages_text if page)
                
                # Add non-text elements to metadata
                metadata['non_text_elements'] = non_text_elements
                
                # Add extraction stats
                metadata['extraction_stats'] = {
                    'total_pages': len(pdf.pages),
                    'processed_pages': len(pages_text) - len(self.exclude_pages),
                    'excluded_pages': len(self.exclude_pages),
                    'tables_count': len(non_text_elements['tables']),
                    'diagrams_count': len(non_text_elements['diagrams']),
                    'formulas_count': len(non_text_elements['formulas'])
                }
                    
            return extracted_text, metadata

        except FileNotFoundError:
             logger.error(f"PDF file not found: {file_path}")
             raise # Re-raise FileNotFoundError or handle as appropriate
        # Add handling for other potential OS errors if needed
        except OSError as e:
            logger.error(f"OS error opening PDF {file_path}: {e}")
            raise # Re-raise or handle as appropriate
        except Exception as e: # Keep the general exception handler
            logger.error(f"Error extracting text from PDF: {file_path}. Error: {str(e)}")
            raise
    
    def _extract_metadata(self, pdf: PdfReader) -> Dict:
        """
        Extract metadata from a PDF document.
        
        Args:
            pdf: PyPDF2 PdfReader object
            
        Returns:
            Dict: Extracted metadata
        """
        metadata = {}
        info = pdf.metadata
        
        if info:
            # Extract standard metadata fields
            metadata_fields = {
                'title': info.get('/Title', ''),
                'author': info.get('/Author', ''),
                'subject': info.get('/Subject', ''),
                'creator': info.get('/Creator', ''),
                'producer': info.get('/Producer', ''),
                'creation_date': info.get('/CreationDate', ''),
                'modification_date': info.get('/ModDate', '')
            }
            
            # Clean up metadata and add to dict
            for key, value in metadata_fields.items():
                if value:
                    # Convert PyPDF2 string objects if needed
                    if isinstance(value, (pypdf.generic.IndirectObject,
                                          pypdf.generic.TextStringObject,
                                          pypdf.generic.ByteStringObject)):
                        metadata[key] = str(value)
                    else:
                        metadata[key] = value
        
        # Add document statistics
        metadata['page_count'] = len(pdf.pages)
        
        return metadata
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Full document processing including text extraction and non-text element handling.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict[str, Any]: Processed document with text and elements
        """
        # Extract text and metadata
        text, metadata = self.extract_text(file_path)
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Create result object
        result = {
            'text': cleaned_text,
            'metadata': metadata,
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'non_text_elements': metadata.get('non_text_elements', {})
        }
        
        # Convert tables to text format for inclusion in the document
        tables_text = []
        for table in result['non_text_elements'].get('tables', []):
            if 'markdown' in table:
                tables_text.append(f"TABLE (Page {table['page'] + 1}):\n{table['markdown']}\n")
        
        # Convert diagrams to text descriptions
        diagrams_text = []
        for diagram in result['non_text_elements'].get('diagrams', []):
            if diagram.get('description'):
                diagrams_text.append(
                    f"DIAGRAM (Page {diagram['page'] + 1}):\n{diagram['description']}\n"
                )
        
        # Convert formulas to text
        formulas_text = []
        for formula in result['non_text_elements'].get('formulas', []):
            description = self.formula_extractor.generate_formula_description(formula)
            formulas_text.append(
                f"FORMULA (Page {formula['page'] + 1}):\n{formula['formula']}\n{description}\n"
            )
        
        # Add non-text elements as text to preserve in knowledge graph
        if tables_text or diagrams_text or formulas_text:
            result['elements_text'] = "\n\n".join(tables_text + diagrams_text + formulas_text)
            # Optionally append this to the main text
            result['text_with_elements'] = cleaned_text + "\n\n" + result['elements_text']
        else:
            result['elements_text'] = ""
            result['text_with_elements'] = cleaned_text
        
        return result
