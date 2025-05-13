"""
Advanced PDF parsing module for LightRAG.

This module provides robust PDF text extraction that preserves document structure
using PyMuPDF (fitz), along with metadata extraction from both PDF content and
file system attributes. It also includes content filtering to remove non-RAG useful
elements like headers, footers, page numbers, and TOC/Index entries. Additionally,
it can extract tables from PDFs and convert them to Markdown format.
"""
import os
import logging
import datetime
from typing import Optional, Dict, Any, Tuple, List

# Import PyMuPDF
import fitz

# Check for pdfplumber availability (used for table extraction)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.getLogger(__name__).warning("pdfplumber library not available. Table extraction will be limited.")

logger = logging.getLogger(__name__)

def extract_structured_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF document while preserving structure using PyMuPDF.

    This function extracts text from a PDF file in a way that preserves the
    document's structure, including paragraphs, columns, and text flow.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text as a single string with structure preserved

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If the file is not a valid PDF
        Exception: For other errors during extraction
    """
    logger.debug(f"Extracting structured text from PDF: {pdf_path}")

    # Validate file existence
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate file extension
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {pdf_path}")
        raise ValueError(f"File is not a PDF: {pdf_path}")

    extracted_text = ""

    try:
        # Open the PDF with PyMuPDF
        with fitz.open(pdf_path) as pdf_document:
            # Get the number of pages
            num_pages = len(pdf_document)
            logger.info(f"Processing PDF with {num_pages} pages")

            # Extract text from each page
            page_texts = []
            for page_num, page in enumerate(pdf_document):
                # Extract text with PyMuPDF's structured text extraction
                # This preserves the reading order and structure better than simple text extraction
                page_text = page.get_text("text")

                if page_text.strip():  # Only add non-empty pages
                    page_texts.append(page_text)
                else:
                    logger.debug(f"Page {page_num + 1} appears to be empty or contains only images")

            # Join all page texts with double newlines to separate pages
            extracted_text = "\n\n".join(page_texts)

            logger.info(f"Successfully extracted text from {num_pages} pages")

    except fitz.FileDataError as e:
        logger.error(f"Invalid or corrupted PDF file: {pdf_path}. Error: {str(e)}")
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}. Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {pdf_path}. Error: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {pdf_path}. Error: {str(e)}")

    return extracted_text


def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document using PyMuPDF.

    This function extracts standard PDF metadata such as title, author, creation date,
    as well as document statistics like page count.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        Dict[str, Any]: Dictionary containing PDF metadata

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If the file is not a valid PDF
        Exception: For other errors during extraction
    """
    logger.debug(f"Extracting metadata from PDF: {pdf_path}")

    # Validate file existence
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate file extension
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {pdf_path}")
        raise ValueError(f"File is not a PDF: {pdf_path}")

    metadata = {}

    try:
        # Open the PDF with PyMuPDF
        with fitz.open(pdf_path) as pdf_document:
            # Get basic document info
            info = pdf_document.metadata

            # Extract standard metadata fields if available
            if info:
                # Map PyMuPDF metadata keys to our standardized keys
                metadata_mapping = {
                    'title': info.get('title', ''),
                    'author': info.get('author', ''),
                    'subject': info.get('subject', ''),
                    'keywords': info.get('keywords', ''),
                    'creator': info.get('creator', ''),
                    'producer': info.get('producer', ''),
                    'creation_date': info.get('creationDate', ''),
                    'modification_date': info.get('modDate', '')
                }

                # Clean up and normalize dates to ISO 8601 format
                for key, value in metadata_mapping.items():
                    if value:
                        # Handle date fields specially
                        if key in ['creation_date', 'modification_date'] and value:
                            try:
                                # PyMuPDF returns dates in format like "D:20201231235959+00'00'"
                                # Convert to ISO 8601 format if possible
                                if value.startswith('D:'):
                                    # Extract components from PyMuPDF date format
                                    date_str = value[2:]  # Remove 'D:' prefix
                                    year = int(date_str[0:4])
                                    month = int(date_str[4:6])
                                    day = int(date_str[6:8])

                                    # Check if time components are present
                                    if len(date_str) >= 14:
                                        hour = int(date_str[8:10])
                                        minute = int(date_str[10:12])
                                        second = int(date_str[12:14])

                                        # Create datetime object and convert to ISO format
                                        dt = datetime.datetime(year, month, day, hour, minute, second)
                                        metadata[key] = dt.isoformat()
                                    else:
                                        # Date only
                                        dt = datetime.date(year, month, day)
                                        metadata[key] = dt.isoformat()
                                else:
                                    # If not in expected format, store as is
                                    metadata[key] = value
                            except (ValueError, IndexError) as e:
                                # If date parsing fails, store original value
                                logger.warning(f"Could not parse date '{value}': {str(e)}")
                                metadata[key] = value
                        else:
                            # For non-date fields, store as is
                            metadata[key] = value

            # Add document statistics
            metadata['page_count'] = len(pdf_document)

            # Get page sizes (first page as reference)
            if len(pdf_document) > 0:
                first_page = pdf_document[0]
                metadata['page_width'] = first_page.rect.width
                metadata['page_height'] = first_page.rect.height

            logger.info(f"Successfully extracted metadata from PDF: {pdf_path}")

    except fitz.FileDataError as e:
        logger.error(f"Invalid or corrupted PDF file: {pdf_path}. Error: {str(e)}")
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}. Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting metadata from PDF: {pdf_path}. Error: {str(e)}")
        raise Exception(f"Error extracting metadata from PDF: {pdf_path}. Error: {str(e)}")

    return metadata


def extract_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract file system metadata for a file.

    This function extracts file system metadata such as file size, creation time,
    modification time, and access time.

    Args:
        file_path (str): Path to the file

    Returns:
        Dict[str, Any]: Dictionary containing file system metadata

    Raises:
        FileNotFoundError: If the file does not exist
    """
    logger.debug(f"Extracting file system metadata for: {file_path}")

    # Validate file existence
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = {}

    try:
        # Get file stats
        file_stats = os.stat(file_path)

        # Extract file system metadata
        metadata['file_size'] = file_stats.st_size  # Size in bytes
        metadata['file_name'] = os.path.basename(file_path)
        metadata['file_extension'] = os.path.splitext(file_path)[1].lower()
        metadata['file_path'] = os.path.abspath(file_path)
        metadata['file_directory'] = os.path.dirname(os.path.abspath(file_path))

        # Convert timestamps to ISO 8601 format
        metadata['creation_time'] = datetime.datetime.fromtimestamp(
            file_stats.st_ctime).isoformat()
        metadata['modification_time'] = datetime.datetime.fromtimestamp(
            file_stats.st_mtime).isoformat()
        metadata['access_time'] = datetime.datetime.fromtimestamp(
            file_stats.st_atime).isoformat()

        logger.info(f"Successfully extracted file system metadata for: {file_path}")

    except Exception as e:
        # Just log the error and re-raise the original exception
        # This makes it easier to test
        logger.error(f"Error extracting file system metadata: {file_path}. Error: {str(e)}")
        raise

    return metadata


def process_pdf_document(pdf_path: str, filter_content: bool = True, extract_tables: bool = True,
                   extract_diagrams: bool = True, extract_formulas: bool = True,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a PDF document to extract text content, metadata, and non-text elements.

    This function combines text extraction, metadata extraction, and extraction of non-text
    elements like tables, diagrams, and formulas. It also applies content filtering to remove
    headers, footers, page numbers, TOC/Index entries, etc. Diagrams and formulas are replaced
    with unique placeholders in the text content.

    Args:
        pdf_path (str): Path to the PDF file
        filter_content (bool, optional): Whether to filter out non-RAG useful content. Defaults to True.
        extract_tables (bool, optional): Whether to extract tables from the PDF. Defaults to True.
        extract_diagrams (bool, optional): Whether to extract diagrams from the PDF. Defaults to True.
        extract_formulas (bool, optional): Whether to extract formulas from the text. Defaults to True.
        context (Optional[Dict[str, Any]], optional): Additional context for advanced processing.
            This can include:
            - schema_validator: SchemaValidator instance for diagram entity extraction
            - llm_func: Async function to call an LLM for diagram entity extraction

    Returns:
        Dict[str, Any]: Dictionary containing text content, metadata, and extracted elements

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If the file is not a valid PDF
        Exception: For other errors during processing
    """
    logger.info(f"Processing PDF document: {pdf_path}")

    result = {
        'extracted_elements': {},
        'context': context or {}
    }

    try:
        # Extract text content
        text_content = extract_structured_text_from_pdf(pdf_path)
        original_text_content = text_content

        # Extract PDF metadata
        pdf_metadata = extract_pdf_metadata(pdf_path)

        # Extract file system metadata
        file_metadata = extract_file_metadata(pdf_path)

        # Initialize placeholders dictionary to track replacements
        placeholders = {}

        # Extract tables if enabled
        if extract_tables:
            try:
                from .table_extractor import extract_tables_with_metadata
                tables = extract_tables_with_metadata(pdf_path)

                if tables:
                    # Add tables to extracted elements
                    result['extracted_elements']['tables'] = tables

                    # Add table count to metadata
                    pdf_metadata['table_count'] = len(tables)

                    logger.info(f"Added {len(tables)} tables to extracted elements")

                    # Replace tables with placeholders in the text
                    # This is optional since tables are already in text format
                    # We'll keep them in the text but also store them separately
            except ImportError:
                logger.warning("Table extraction module not available. Tables will not be extracted.")
            except Exception as e:
                logger.error(f"Error extracting tables: {str(e)}")
                # Continue processing even if table extraction fails

        # Extract diagrams if enabled
        if extract_diagrams:
            try:
                from .diagram_analyzer import DiagramAnalyzer, PYMUPDF_AVAILABLE

                if PYMUPDF_AVAILABLE:
                    # Initialize diagram analyzer
                    diagram_analyzer = DiagramAnalyzer()

                    # Extract diagrams
                    diagrams = diagram_analyzer.extract_diagrams_from_pdf(pdf_path)

                    if diagrams:
                        # Add diagrams to extracted elements
                        result['extracted_elements']['diagrams'] = diagrams

                        # Add diagram count to metadata
                        pdf_metadata['diagram_count'] = len(diagrams)

                        logger.info(f"Added {len(diagrams)} diagrams to extracted elements")

                        # Replace diagrams with placeholders in the text
                        for diagram in diagrams:
                            if diagram.get('position') and diagram.get('page'):
                                # Generate a unique placeholder
                                placeholder = f"[DIAGRAM-{diagram['diagram_id']}]"

                                # Store the placeholder for reference
                                placeholders[placeholder] = {
                                    'type': 'diagram',
                                    'id': diagram['diagram_id']
                                }

                                # Generate a basic description if none exists
                                if not diagram.get('description'):
                                    diagram['description'] = f"Diagram on page {diagram['page']}"
                                    if diagram.get('caption'):
                                        diagram['description'] += f": {diagram['caption']}"

                                # For diagrams, we'll add placeholders in the filtered text later
                                # since we don't have exact text positions to replace

                        # Extract entities and relationships from diagrams if schema and LLM provided
                        try:
                            from .diagram_entity_extractor import DiagramEntityExtractor

                            # Check if schema and LLM are available via the context
                            schema_validator = result.get('context', {}).get('schema_validator')
                            llm_func = result.get('context', {}).get('llm_func')

                            if schema_validator and llm_func:
                                logger.info("Extracting entities and relationships from diagrams")

                                # Initialize diagram entities and relationships lists
                                all_diagram_entities = []
                                all_diagram_relationships = []

                                # Process each diagram
                                import asyncio

                                # Create event loop
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)

                                # Process diagrams one by one
                                for diagram in diagrams:
                                    # Generate description for diagram if not already present
                                    if not diagram.get('description'):
                                        diagram['description'] = loop.run_until_complete(
                                            diagram_analyzer.generate_diagram_description(diagram)
                                        )

                                    # Extract entities and relationships
                                    entities, relationships = loop.run_until_complete(
                                        diagram_analyzer.extract_entities_and_relationships(
                                            diagram, schema_validator, llm_func
                                        )
                                    )

                                    # Add to lists
                                    all_diagram_entities.extend(entities)
                                    all_diagram_relationships.extend(relationships)

                                # Add results to extracted elements
                                if all_diagram_entities:
                                    result['extracted_elements']['diagram_entities'] = all_diagram_entities
                                    pdf_metadata['diagram_entity_count'] = len(all_diagram_entities)
                                    logger.info(f"Added {len(all_diagram_entities)} entities from diagrams")

                                if all_diagram_relationships:
                                    result['extracted_elements']['diagram_relationships'] = all_diagram_relationships
                                    pdf_metadata['diagram_relationship_count'] = len(all_diagram_relationships)
                                    logger.info(f"Added {len(all_diagram_relationships)} relationships from diagrams")
                            else:
                                logger.debug("Schema validator or LLM function not provided. Skipping diagram entity extraction.")
                        except ImportError:
                            logger.warning("Diagram entity extractor module not available. Diagram entities will not be extracted.")
                        except Exception as e:
                            logger.error(f"Error extracting entities from diagrams: {str(e)}")
                            # Continue processing even if entity extraction fails
                else:
                    logger.warning("PyMuPDF not available. Diagram extraction skipped.")
            except ImportError:
                logger.warning("Diagram analyzer module not available. Diagrams will not be extracted.")
            except Exception as e:
                logger.error(f"Error extracting diagrams: {str(e)}")
                # Continue processing even if diagram extraction fails

        # Extract formulas if enabled
        if extract_formulas:
            try:
                from .formula_extractor import FormulaExtractor

                # Initialize formula extractor
                formula_extractor = FormulaExtractor()

                # Extract formulas from text
                formulas = formula_extractor.extract_formulas(text_content)

                if formulas:
                    # Add formulas to extracted elements
                    result['extracted_elements']['formulas'] = formulas

                    # Add formula count to metadata
                    pdf_metadata['formula_count'] = len(formulas)

                    logger.info(f"Added {len(formulas)} formulas to extracted elements")

                    # Replace formulas with placeholders in the text
                    # We need to replace from end to start to avoid position shifts
                    for formula in sorted(formulas, key=lambda x: x['position'][0], reverse=True):
                        start, end = formula['position']
                        formula_text = formula['formula']

                        # Generate a unique placeholder
                        placeholder = f"[FORMULA-{formula['formula_id']}]"

                        # Store the placeholder for reference
                        placeholders[placeholder] = {
                            'type': 'formula',
                            'id': formula['formula_id']
                        }

                        # Generate a description if none exists
                        if not formula.get('description'):
                            formula['description'] = formula_extractor.generate_formula_description(formula)

                        # Replace the formula with the placeholder in the text
                        text_content = text_content[:start] + placeholder + text_content[end:]
            except ImportError:
                logger.warning("Formula extractor module not available. Formulas will not be extracted.")
            except Exception as e:
                logger.error(f"Error extracting formulas: {str(e)}")
                # Continue processing even if formula extraction fails

        # Apply content filtering if enabled
        if filter_content:
            from .content_filter import filter_extracted_text
            filtered_text = filter_extracted_text(text_content)
            result['text_content'] = filtered_text
            result['original_text_content'] = original_text_content  # Keep original for reference
        else:
            result['text_content'] = text_content

        # Add diagram placeholders to filtered text if diagrams were extracted
        if extract_diagrams and 'diagrams' in result['extracted_elements']:
            # For each diagram, find a suitable position in the filtered text
            # This is a heuristic approach since we don't have exact text positions after filtering
            for diagram in result['extracted_elements']['diagrams']:
                if diagram.get('page'):
                    # Generate a unique placeholder
                    placeholder = f"[DIAGRAM-{diagram['diagram_id']}]"

                    # Try to find a good position based on surrounding text
                    if diagram.get('surrounding_text') and diagram['surrounding_text'] in result['text_content']:
                        # If surrounding text is found, insert after it
                        pos = result['text_content'].find(diagram['surrounding_text']) + len(diagram['surrounding_text'])
                        result['text_content'] = result['text_content'][:pos] + "\n" + placeholder + "\n" + result['text_content'][pos:]
                    elif diagram.get('caption') and diagram['caption'] in result['text_content']:
                        # If caption is found, insert after it
                        pos = result['text_content'].find(diagram['caption']) + len(diagram['caption'])
                        result['text_content'] = result['text_content'][:pos] + "\n" + placeholder + "\n" + result['text_content'][pos:]
                    else:
                        # Otherwise, add at the end of the document
                        result['text_content'] += f"\n\n{placeholder}\n"

        # Store placeholders in the result
        result['placeholders'] = placeholders

        # Combine metadata
        result['metadata'] = {
            **pdf_metadata,
            **file_metadata,
            'extraction_timestamp': datetime.datetime.now().isoformat(),
            'content_filtered': filter_content,
            'placeholders_count': len(placeholders)
        }

        logger.info(f"Successfully processed PDF document: {pdf_path}")

    except Exception as e:
        logger.error(f"Error processing PDF document: {pdf_path}. Error: {str(e)}")
        raise

    return result
