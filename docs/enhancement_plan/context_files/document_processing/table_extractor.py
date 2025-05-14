# /home/sergeblumenfeld/graphrag-tutor/graphrag_tutor/document_processing/table_extractor.py
"""
Table extraction and conversion module for GraphRAG tutor.
"""
import logging
import os
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import io

logger = logging.getLogger(__name__)


# Try to import table extraction libraries with graceful fallbacks
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("Camelot library not available. Installing it is recommended for better table extraction.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber library not available. Installing it is recommended as a fallback for table extraction.")

try:
    import pypdf
    PYPDF2_AVAILABLE = True # Keep variable name for now, but it reflects pypdf availability
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("pypdf library not available. Basic PDF parsing will be limited.")

class TableExtractor:
    """
    Extract and convert tables from PDF documents to Markdown.
    """
    def __init__(self, preferred_engine: str = 'auto'):
        """
        Initialize the table extractor.

        Args:
            preferred_engine: Preferred extraction engine ('camelot', 'pdfplumber', or 'auto')
        """
        self.preferred_engine = preferred_engine

        # Determine available engines
        self.available_engines = []
        if CAMELOT_AVAILABLE:
            self.available_engines.append('camelot')
        if PDFPLUMBER_AVAILABLE:
            self.available_engines.append('pdfplumber')

        if not self.available_engines:
            logger.warning("No table extraction engines available. Tables will not be properly extracted.")
        else:
            logger.info(f"Table extraction using engines: {', '.join(self.available_engines)}")

    def _get_engine(self) -> str:
        """
        Get the best available extraction engine.

        Returns:
            str: Name of the extraction engine to use
        """
        if self.preferred_engine != 'auto' and self.preferred_engine in self.available_engines:
            return self.preferred_engine

        # Auto-select best available
        if 'camelot' in self.available_engines:
            return 'camelot'
        elif 'pdfplumber' in self.available_engines:
            return 'pdfplumber'
        else:
            return 'none'

    def _extract_with_camelot(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Extract tables using Camelot.

        Args:
            pdf_path: Path to the PDF file
            pages: Specific pages to extract tables from

        Returns:
            List[Dict]: Extracted tables
        """
        if not CAMELOT_AVAILABLE:
            return []

        try:
            # Convert page numbers to string format required by camelot
            page_str = None
            if pages:
                # Camelot uses 1-indexed pages
                page_str = ','.join(str(p + 1) for p in pages)

            # Extract tables - try both lattice and stream methods
            tables = []

            # Try lattice mode (for tables with borders)
            lattice_tables = camelot.read_pdf(
                pdf_path,
                pages=page_str,
                flavor='lattice'
            )

            # Try stream mode (for tables without clear borders)
            stream_tables = camelot.read_pdf(
                pdf_path,
                pages=page_str,
                flavor='stream'
            )

            # Process lattice tables
            for i, table in enumerate(lattice_tables):
                if table.df.empty:
                    continue

                table_data = table.df.values.tolist()
                header = table.df.columns.tolist()

                # Insert header as first row for consistent processing
                table_data.insert(0, header)

                # Get page number (0-indexed for consistency)
                page_num = table.page - 1

                tables.append({
                    'table_id': f'lattice-{page_num}-{i}',
                    'page': page_num,
                    'data': table_data,
                    'accuracy': table.accuracy,
                    'extraction_method': 'camelot-lattice',
                    'markdown': self.convert_table_to_markdown(table_data)
                })

            # Process stream tables, favoring lattice when we have high-confidence tables
            for i, table in enumerate(stream_tables):
                if table.df.empty:
                    continue

                # Check if we already have a high-confidence lattice table for this area
                page_num = table.page - 1

                # Skip if we already have a good lattice table for this page
                existing = [t for t in tables if t['page'] == page_num and t['accuracy'] > 90]
                if existing:
                    continue

                table_data = table.df.values.tolist()
                header = table.df.columns.tolist()
                table_data.insert(0, header)

                tables.append({
                    'table_id': f'stream-{page_num}-{i}',
                    'page': page_num,
                    'data': table_data,
                    'accuracy': table.accuracy,
                    'extraction_method': 'camelot-stream',
                    'markdown': self.convert_table_to_markdown(table_data)
                })

            return tables

        except Exception as e:
            logger.error(f"Error extracting tables with camelot: {str(e)}")
            return []

    def _extract_with_pdfplumber(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber.

        Args:
            pdf_path: Path to the PDF file
            pages: Specific pages to extract tables from

        Returns:
            List[Dict]: Extracted tables
        """
        if not PDFPLUMBER_AVAILABLE:
            return []

        try:
            tables = []

            with pdfplumber.open(pdf_path) as pdf:
                # Determine which pages to process
                page_indices = pages if pages is not None else range(len(pdf.pages))

                for page_idx in page_indices:
                    if page_idx >= len(pdf.pages):
                        continue

                    page = pdf.pages[page_idx]

                    # Extract tables from the page
                    page_tables = page.extract_tables()

                    # Process each table
                    for i, table_data in enumerate(page_tables):
                        if not table_data or not any(table_data):
                            continue

                        # Clean up table data - replace None with empty string
                        cleaned_data = [[cell if cell is not None else '' for cell in row] for row in table_data]

                        # Create table object
                        tables.append({
                            'table_id': f'pdfplumber-{page_idx}-{i}',
                            'page': page_idx,
                            'data': cleaned_data,
                            'accuracy': 70,  # Assume moderate accuracy for pdfplumber
                            'extraction_method': 'pdfplumber',
                            'markdown': self.convert_table_to_markdown(cleaned_data)
                        })

            return tables

        except Exception as e:
            logger.error(f"Error extracting tables with pdfplumber: {str(e)}")
            return []

    def _normalize_table(self, table_data: List[List[str]]) -> List[List[str]]:
        """
        Normalize table data for consistent processing.

        Args:
            table_data: Raw table data

        Returns:
            List[List[str]]: Normalized table data
        """
        if not table_data:
            return []

        # Ensure all rows have the same number of columns
        # Filter out potential None rows before calculating max_cols
        valid_rows = [row for row in table_data if isinstance(row, list)]
        if not valid_rows:
            return []
        max_cols = max(len(row) for row in valid_rows)
        normalized = []

        for row in valid_rows:
            # Pad rows with empty strings if needed
            padded_row = row + [''] * (max_cols - len(row))
            normalized.append(padded_row)

        return normalized

    def extract_tables_from_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            pages: Specific pages to extract tables from (if None, extracts from all pages)

        Returns:
            List[Dict]: List of extracted tables with metadata
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        # Get the best available engine
        engine = self._get_engine()
        logger.info(f"Extracting tables from {pdf_path} using {engine} engine")

        # Extract tables with selected engine
        tables = []

        if engine == 'camelot':
            tables = self._extract_with_camelot(pdf_path, pages)
        elif engine == 'pdfplumber':
            tables = self._extract_with_pdfplumber(pdf_path, pages)

        # If no tables were found, try the fallback engine
        if not tables and 'camelot' in self.available_engines and engine != 'camelot':
            logger.info("No tables found with primary engine, trying camelot")
            tables = self._extract_with_camelot(pdf_path, pages)
        elif not tables and 'pdfplumber' in self.available_engines and engine != 'pdfplumber':
            logger.info("No tables found with primary engine, trying pdfplumber")
            tables = self._extract_with_pdfplumber(pdf_path, pages)

        # Post-process and deduplicate tables
        result = []
        seen_content = set()

        for table in tables:
            # Create a fingerprint of the table content
            table_text = ''.join(''.join(str(cell) for cell in row) for row in table['data'])
            fingerprint = hash(table_text)

            # Skip if we've seen this table before
            if fingerprint in seen_content:
                continue

            seen_content.add(fingerprint)
            result.append(table)

        logger.info(f"Extracted {len(result)} unique tables from PDF")
        return result

    def convert_table_to_markdown(self, table_data: List[List[str]]) -> str:
        """
        Convert a table to Markdown format. Includes further refined header detection
        and a more robust fallback mechanism.

        Args:
            table_data: 2D array representing the table

        Returns:
            str: Markdown representation of the table
        """
        if not table_data: # Handle empty list input
            return ""
        # Ensure it's a list of lists and not just an empty list containing nothing or empty lists
        if not any(isinstance(row, list) and row for row in table_data):
             return ""

        try:
            # Normalize the table data
            normalized_data = self._normalize_table(table_data)
            if not normalized_data:
                logger.warning("Table data became empty after normalization.")
                return ""
            logger.debug(f"Normalized data (first 2 rows): {normalized_data[:2]}")

            # --- Stricter Header Detection Heuristic ---
            has_header = False # Default to False
            if len(normalized_data) > 1:
                first_row = normalized_data[0]
                # 1. Basic validity: all cells are non-empty strings?
                is_valid_format = all(isinstance(cell, str) and cell for cell in first_row)
                # 2. Check if any cell in the first row is purely numeric (strong indicator it's NOT a header)
                # Use isdigit() for integer check, handle potential floats/decimals if needed later
                first_row_has_numeric_cell = any(cell.isdigit() for cell in first_row)

                if is_valid_format and not first_row_has_numeric_cell:
                    # If format is valid and no cell is purely numeric, tentatively consider it a header.
                    # Add more checks here if needed (e.g., compare with second row)
                    has_header = True
                    logger.debug("Header tentatively detected based on valid format and no purely numeric cells.")
                elif first_row_has_numeric_cell:
                    logger.debug("First row contains numeric cells, likely not a header.")
                else:
                    logger.debug("First row format invalid for header (non-string or empty cells).")
            else:
                logger.debug("Table has less than 2 rows, cannot reliably detect header.")
            # --- End of Heuristic ---

            df = None
            markdown_table = ""

            if has_header:
                logger.debug("Attempting Markdown conversion with detected header.")
                header = normalized_data[0]
                data_rows = normalized_data[1:]
                if header and data_rows:
                     try:
                         df = pd.DataFrame(data_rows, columns=header)
                         df.columns = [str(col) for col in df.columns] # Clean column names
                         markdown_table = df.to_markdown(index=False, headers="keys")
                         logger.debug("Markdown with header generated successfully.")
                         return markdown_table
                     except Exception as e_header:
                         logger.warning(f"Pandas conversion with detected header failed: {e_header}. Falling back.")
                         has_header = False # Force fallback
                else:
                     logger.warning("Header detected, but header or data rows are empty. Falling back.")
                     has_header = False # Force fallback

            # Fallback: No header detected OR header conversion failed
            if not has_header:
                 logger.warning("No clear header found for table, attempting fallback markdown conversion (without header row).")
                 try:
                     # Create DataFrame without assuming headers
                     logger.debug(f"Fallback: Creating DataFrame from normalized data: {normalized_data}")
                     df = pd.DataFrame(normalized_data)
                     if df.empty:
                         logger.warning("Fallback: DataFrame created from normalized data is empty.")
                         return ""

                     # Attempt pandas markdown conversion first (within the fallback)
                     try:
                         markdown_table = df.to_markdown(index=False, header=False)
                         logger.debug("Fallback: Pandas Markdown (no header) generated successfully.")
                         return markdown_table
                     except Exception as e_to_markdown:
                         # If pandas to_markdown fails, try manual generation
                         logger.warning(f"Fallback: df.to_markdown(header=False) failed: {e_to_markdown}. Attempting manual markdown generation.")

                         manual_markdown_rows = []
                         for row in normalized_data:
                             # Ensure all cells are strings and add padding/stripping
                             formatted_row = [str(cell).strip() for cell in row]
                             manual_markdown_rows.append("| " + " | ".join(formatted_row) + " |")

                         if not manual_markdown_rows:
                              logger.warning("Fallback: Manual markdown generation resulted in no rows.")
                              return ""

                         markdown_table = "\n".join(manual_markdown_rows)
                         logger.debug("Fallback: Manual Markdown generated successfully.")
                         return markdown_table

                 except Exception as e_fallback_df:
                     # This catches errors during DataFrame creation itself
                     logger.error(f"Fallback: Error creating DataFrame: {e_fallback_df}", exc_info=True)
                     return "" # Return empty on error during DataFrame creation in fallback

        except Exception as e:
            # This catches errors before or during normalization, or other unexpected issues
            logger.error(f"Overall error converting table to Markdown: {str(e)}", exc_info=True)
            return ""

        # Should not be reached
        return ""

    def table_data_to_df(self, table_data: List[List[str]]) -> Optional[pd.DataFrame]:
        """
        Convert table data to pandas DataFrame.

        Args:
            table_data: 2D array representing the table

        Returns:
            pd.DataFrame: DataFrame version of the table
        """
        if not table_data or not table_data[0]:
            return None

        try:
            # Normalize table data
            normalized_data = self._normalize_table(table_data)

            # Assume first row is header
            df = pd.DataFrame(normalized_data[1:], columns=normalized_data[0])
            return df
        except Exception as e:
            logger.error(f"Error converting table to DataFrame: {str(e)}")
            return None
