Refactor the PDF parsing functionality within the LightRAG project based on the following specifications.

**Goal:** Implement a robust `PDFParser` class capable of reliably extracting text content page-by-page from PDF documents, handling common errors, providing basic metadata, and preparing the output for subsequent content filtering steps.

**Target Files:**

1.  `lightrag/exceptions.py`: Define custom exceptions here.
2.  `lightrag/document_processing/pdf_parser.py`: Implement the `PDFParser` class here. (If this file doesn't exist, create it. If similar functionality exists elsewhere, instruct aider to update that file instead, adjusting the class/method names as needed).

**Detailed Implementation Steps:**

1.  **Define Custom Exception (in `lightrag/exceptions.py`):**
    *   Ensure the following custom exception class is defined or added:
        ```python
        class PDFReadError(Exception):
            """Custom exception for PDF reading errors (e.g., corrupted file)."""
            pass
        ```

2.  **Implement `PDFParser` Class (in `lightrag/document_processing/pdf_parser.py`):**
    *   **Imports:** Add necessary imports:
        ```python
        import os
        import logging
        from typing import List, Dict, Tuple, Any
        import fitz  # PyMuPDF
        from lightrag.exceptions import PDFReadError # Assuming exceptions are in this path
        ```
    *   **Logging:** Initialize a logger at the module level:
        ```python
        logger = logging.getLogger(__name__)
        ```
    *   **Class Definition:** Define the `PDFParser` class.
        ```python
        class PDFParser:
            """Parses PDF documents to extract text and basic metadata."""
            def __init__(self):
                 """Initializes the PDFParser."""
                 # Add any initialization if needed in the future, otherwise pass
                 pass

            # Implement methods below within this class
        ```
    *   **`validate_pdf(self, file_path: str) -> bool` Method:**
        *   Add a docstring explaining its purpose.
        *   Check if `file_path` exists using `os.path.exists`. Log `ERROR` and return `False` if not found.
        *   Check if the file extension is `.pdf` (case-insensitive). Log `ERROR` and return `False` if not.
        *   Use a `try...except` block to attempt opening the file with `fitz.open(file_path)`.
        *   Catch `fitz.FitzError` (or a more specific fitz exception related to reading errors if applicable) and general `Exception`.
        *   If an exception occurs during opening, log an `ERROR` (e.g., "Invalid or corrupted PDF: {file_path}, Error: {e}") and return `False`.
        *   If opening is successful (no exception), ensure the document is closed within the `try` block (e.g., using `with fitz.open(...) as doc:` or `doc.close()` in `finally`), and return `True`.

    *   **`extract_text(self, file_path: str) -> Tuple[str, List[str], Dict[str, Any]]` Method:**
        *   Add a docstring explaining its purpose and return values.
        *   Log the start of processing at `DEBUG` level: `logger.debug(f"Processing PDF: {file_path}")`.
        *   Call `self.validate_pdf(file_path)`. If it returns `False`, raise `ValueError(f"Invalid or non-existent PDF file: {file_path}")`.
        *   Use a `try...except FileNotFoundError` block around `fitz.open`. Log `ERROR` and re-raise.
        *   Use a `try...except fitz.FitzError as e:` block around `fitz.open`. Log `ERROR` and raise `PDFReadError(f"Corrupted or unreadable PDF: {file_path}: {e}") from e`.
        *   Use a `try...finally` block or `with fitz.open(file_path) as doc:` to ensure the `fitz.Document` object is closed.
        *   **Inside the `try` (or `with`) block:**
            *   Initialize an empty list: `page_texts: List[str] = []`.
            *   Get basic PDF metadata: `metadata: Dict[str, Any] = doc.metadata or {}`. Add page count: `metadata['page_count'] = len(doc)`. Clean metadata values (e.g., remove PDF specific encoding if present).
            *   Iterate through pages: `for i, page in enumerate(doc):`.
            *   Inside the loop, use a nested `try...except Exception as page_error:` block around `page.get_text("text")`.
            *   If `page.get_text()` succeeds, append the extracted `page_text` to `page_texts`.
            *   If an exception occurs during page extraction, log a `WARNING` (e.g., "Could not extract text from page {i+1} in {file_path}: {page_error}"), and append an empty string `""` to `page_texts`. **Crucially, continue to the next page.**
            *   After the loop, log the number of pages processed: `logger.info(f"Successfully extracted text from {len(doc)} pages in {file_path}")`.
            *   Create `concatenated_raw_text` by joining `page_texts` with a suitable separator (e.g., `\n\n`). Filter out empty strings from `page_texts` *before* joining if desired for the concatenated version, but return the *full* `page_texts` list including empty strings.
            *   **Return Value:** Return the tuple `(concatenated_raw_text, page_texts, metadata)`.

**Testing Strategy (Guidance for subsequent steps):**

*   **Unit Tests:**
    *   Create `tests/test_pdf_parser.py`.
    *   Mock `os.path.exists` and `fitz.open`.
    *   Test `validate_pdf`:
        *   Case: File doesn't exist (mock `os.path.exists` -> `False`). Assert `False`.
        *   Case: File is not PDF. Assert `False`.
        *   Case: `fitz.open` raises `FitzError` (mock `fitz.open` side effect). Assert `False`.
        *   Case: Valid PDF (mock `fitz.open` success). Assert `True`.
    *   Test `extract_text`:
        *   Case: `validate_pdf` fails (mock `validate_pdf` -> `False`). Assert `ValueError` is raised.
        *   Case: `fitz.open` raises `FileNotFoundError`. Assert `FileNotFoundError`.
        *   Case: `fitz.open` raises `FitzError`. Assert `PDFReadError`.
        *   Case: Mock `fitz.Document` and `fitz.Page`. Simulate successful text extraction for all pages. Verify `concatenated_raw_text` and `page_texts` list are correct. Verify metadata includes `page_count`.
        *   Case: Mock `fitz.Page.get_text` to raise an `Exception` for one page. Verify the corresponding entry in `page_texts` is `""`, `concatenated_raw_text` contains text from other pages, and a `WARNING` was logged.
*   **Integration Tests:**
    *   Create actual test PDF files (e.g., simple text, multiple pages, one corrupted, one non-PDF).
    *   Run `extract_text` on these files.
    *   Assert expected text content (or parts of it) in the output.
    *   Assert `PDFReadError` for corrupted file, `ValueError` for non-PDF.
    *   Check logs for expected messages (INFO, WARNING, ERROR).

**Important Notes for Implementation:**

*   Ensure all imports (`os`, `logging`, `typing`, `fitz`, custom exception) are correctly placed.
*   Adhere strictly to the specified logging levels and messages.
*   The `page_texts` list returned must correspond to the actual pages (including empty strings for pages where extraction failed) so that subsequent filtering steps can rely on the indices.