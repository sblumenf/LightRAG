"""
Table extraction module for LightRAG.

This module provides functionality to extract tables from PDF documents
and convert them to Markdown format for inclusion in the knowledge graph.
"""
import logging
import os
from typing import List, Optional, Dict, Any, Union

# Import pandas for DataFrame operations
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas library not available. Some table operations will be limited.")

# Import pdfplumber for table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber library not available. Install it for table extraction functionality.")

logger = logging.getLogger(__name__)

def extract_tables_to_markdown(pdf_path: str) -> List[str]:
    """
    Extract tables from a PDF file and convert them to Markdown format.

    This function uses pdfplumber to extract tables from a PDF document and
    converts each table to a Markdown string representation.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[str]: List of Markdown strings, each representing a table
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber is not available. Cannot extract tables.")
        return []

    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    markdown_tables = []

    try:
        # Validate the PDF file before opening
        if not os.path.getsize(pdf_path) > 0:
            logger.error(f"PDF file is empty: {pdf_path}")
            return []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Extracting tables from {pdf_path} ({len(pdf.pages)} pages)")

                for page_idx, page in enumerate(pdf.pages):
                    try:
                        # Extract tables from the page
                        tables = page.extract_tables()

                        if not tables:
                            continue

                        logger.debug(f"Found {len(tables)} tables on page {page_idx+1}")

                        for table_idx, table in enumerate(tables):
                            if not table or not any(table):
                                continue

                            # Convert table to Markdown
                            markdown = _table_to_markdown(table)
                            if markdown:
                                markdown_tables.append(markdown)
                                logger.debug(f"Converted table {table_idx+1} on page {page_idx+1} to Markdown")

                    except Exception as e:
                        logger.error(f"Error extracting tables from page {page_idx+1}: {str(e)}")
                        continue

        except ValueError as e:
            if "No /Root object!" in str(e):
                logger.error(f"Invalid PDF structure in {pdf_path}: {str(e)}")
            else:
                logger.error(f"Error opening PDF file for table extraction: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error opening PDF file for table extraction: {str(e)}")
            return []

    except Exception as e:
        logger.error(f"Unexpected error during table extraction: {str(e)}")
        return []

    logger.info(f"Extracted {len(markdown_tables)} tables from {pdf_path}")
    return markdown_tables

def _table_to_markdown(table: List[List[Any]]) -> str:
    """
    Convert a table to Markdown format.

    Args:
        table (List[List[Any]]): 2D array representing the table

    Returns:
        str: Markdown representation of the table
    """
    if not table or not any(table):
        return ""

    # Clean up table data - replace None with empty string
    # Filter out non-list rows
    cleaned_table = []
    for row in table:
        if isinstance(row, (list, tuple)):
            cleaned_table.append([cell if cell is not None else '' for cell in row])

    if not cleaned_table:
        return ""

    # Ensure all rows have the same number of columns

    max_cols = max(len(row) for row in cleaned_table)
    normalized_table = [row + [''] * (max_cols - len(row)) for row in cleaned_table]

    # Create Markdown table
    markdown_rows = []

    # Add header row
    markdown_rows.append("| " + " | ".join(str(cell).strip() for cell in normalized_table[0]) + " |")

    # Add separator row
    markdown_rows.append("| " + " | ".join(["---"] * max_cols) + " |")

    # Add data rows
    for row in normalized_table[1:]:
        markdown_rows.append("| " + " | ".join(str(cell).strip() for cell in row) + " |")

    return "\n".join(markdown_rows)

def extract_tables_with_metadata(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF file with additional metadata.

    This function is similar to extract_tables_to_markdown but returns more
    detailed information about each table, including page number and position.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing table data and metadata
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber is not available. Cannot extract tables.")
        return []

    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    tables_with_metadata = []

    try:
        # Validate the PDF file before opening
        if not os.path.getsize(pdf_path) > 0:
            logger.error(f"PDF file is empty: {pdf_path}")
            return []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Extracting tables with metadata from {pdf_path} ({len(pdf.pages)} pages)")

                for page_idx, page in enumerate(pdf.pages):
                    try:
                        # Extract tables from the page
                        tables = page.extract_tables()

                        # Skip pages with no tables
                        if not tables:
                            continue

                        for table_idx, table in enumerate(tables):
                            if not table or not any(table):
                                continue

                            # Clean up table data - replace None with empty string
                            cleaned_data = [[cell if cell is not None else '' for cell in row] for row in table]

                            # Convert table to Markdown
                            markdown = _table_to_markdown(table)

                            # Skip tables that couldn't be converted to markdown
                            if not markdown:
                                continue

                            # Create table metadata
                            table_info = {
                                'table_id': f"table-{page_idx+1}-{table_idx+1}",
                                'page': page_idx,
                                'page_number': page_idx + 1,
                                'data': cleaned_data,
                                'markdown': markdown,
                                'extraction_method': 'pdfplumber'
                            }

                            tables_with_metadata.append(table_info)

                    except Exception as e:
                        logger.error(f"Error extracting tables from page {page_idx+1}: {str(e)}")
                        continue

        except ValueError as e:
            if "No /Root object!" in str(e):
                logger.error(f"Invalid PDF structure in {pdf_path}: {str(e)}")
            else:
                logger.error(f"Error opening PDF file for table extraction: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error opening PDF file for table extraction: {str(e)}")
            return []

    except Exception as e:
        logger.error(f"Unexpected error during table extraction with metadata: {str(e)}")
        return []

    logger.info(f"Extracted {len(tables_with_metadata)} tables with metadata from {pdf_path}")
    return tables_with_metadata


def table_data_to_df(table_data: List[List[str]]) -> Optional[pd.DataFrame]:
    """
    Convert table data to pandas DataFrame.

    Args:
        table_data: 2D array representing the table

    Returns:
        pd.DataFrame: DataFrame version of the table, or None if conversion fails
    """
    if not table_data or not table_data[0]:
        return None

    try:
        # Normalize table data
        normalized_data = []
        for row in table_data:
            if isinstance(row, list):
                # Replace None with empty string
                normalized_row = [cell if cell is not None else '' for cell in row]
                normalized_data.append(normalized_row)

        if not normalized_data:
            return None

        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in normalized_data)
        for i, row in enumerate(normalized_data):
            if len(row) < max_cols:
                normalized_data[i] = row + [''] * (max_cols - len(row))

        # Assume first row is header
        df = pd.DataFrame(normalized_data[1:], columns=normalized_data[0])
        return df
    except Exception as e:
        logger.error(f"Error converting table to DataFrame: {str(e)}")
        return None
