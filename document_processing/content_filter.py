"""
Content filtering module for LightRAG.

This module provides functionality to filter out non-RAG useful content from documents,
such as headers, footers, page numbers, tables of contents, and indices.
"""
import re
import logging
from typing import List, Dict, Optional, Union, Tuple, Set

logger = logging.getLogger(__name__)

class ContentFilter:
    """
    Filters out non-RAG useful content from document text.
    """
    def __init__(self, language: str = 'en'):
        """
        Initialize the content filter.

        Args:
            language: Language code for language-specific filtering
        """
        self.language = language

        # Common patterns for non-RAG useful content
        self.patterns = {
            # Headers and footers
            'header_footer': [
                r'^\s*Page \d+(?: of \d+)?\s*$',  # Page X or Page X of Y
                r'^\s*\d+\s*$',  # Just page numbers
                r'^\s*[A-Za-z0-9_\- ]+\s*\|\s*[A-Za-z0-9_\- ]+\s*$',  # Common header format
                r'^\s*www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}\s*$',  # Website URLs in headers/footers
                r'^\s*[©Cc]opyright\s+\d{4}.*$',  # Copyright notices
                r'^\s*[Cc]onfidential\s*$',  # Confidentiality notices
            ],

            # Table of contents patterns
            'toc': [
                r'^\s*Table\s+of\s+Contents\s*$',
                r'^\s*Contents\s*$',
                r'^\s*CONTENTS\s*$',
                # Matches lines like "Chapter 1: Title....... 5" or "1.2 Section...10"
                r'^\s*(?:Chapter\s+[\dIVXLCDMivxlcdm]+|[\d\.]+|Appendix\s+[A-Z])[:\.\s].*?[\. ]{3,}\s*\d+\s*$',
                # General pattern for lines ending with dots/spaces and page number
                r'^\s*.*[\. ]{3,}\s*\d+\s*$',
            ],

            # Index patterns
            'index': [
                r'^\s*Index\s*$',
                r'^\s*INDEX\s*$',
                r'^\s*Subject\s+Index\s*$',
                # Matches lines like "Term, 1, 5-10" or "Another Term, 23"
                r'^\s*[A-Za-z][\w\s,-]+,\s*\d+([\s,-]+\d+)*\s*$',
                # General pattern for lines ending with dots/spaces and page number (can also appear in indices)
                r'^\s*.*[\. ]{3,}\s*\d+\s*$',
            ],

            # References and bibliography
            'references': [
                r'^\s*References\s*$',
                r'^\s*REFERENCES\s*$',
                r'^\s*Bibliography\s*$',
                r'^\s*BIBLIOGRAPHY\s*$',
                r'^(\[\d+\]|\d+\.)\s+[A-Z][a-z]+.*(\(\d{4}\)|,\s*\d{4}).*$',  # [1] Author et al. (2020)...
            ],

            # Disclaimers and legal notices
            'disclaimers': [
                r'^\s*Disclaimer:.*$',
                r'^\s*DISCLAIMER:.*$',
                r'^\s*This document is for informational purposes only.*$',
                r'^\s*All rights reserved.*$',
                r'^\s*Terms\s+(and|&)\s+Conditions\s*$',
            ]
        }

    def is_toc_or_index_page(self, text: str, threshold: float = 0.5) -> bool:
        """
        Determine if a page is likely a table of contents or index page.

        Args:
            text: Text content of the page
            threshold: Percentage of lines matching TOC/index patterns to qualify

        Returns:
            bool: True if the page is likely TOC or index
        """
        if not text or len(text.strip()) < 30:  # Skip very short or empty pages
            return False

        lines = text.strip().split('\n')
        # Combine TOC and Index patterns for checking
        toc_index_patterns = self.patterns['toc'] + self.patterns['index']
        compiled_patterns = [re.compile(p) for p in toc_index_patterns]  # Compile for efficiency

        # Count lines matching TOC/index patterns
        matching_lines = 0
        non_empty_lines = 0
        has_explicit_header = False

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            non_empty_lines += 1

            for pattern in compiled_patterns:
                if pattern.match(stripped_line):
                    matching_lines += 1
                    # Check if it's one of the main header lines
                    if stripped_line.lower() in ["table of contents", "contents", "index", "subject index"]:
                        has_explicit_header = True
                    break  # Move to next line once a match is found

        # If there are no non-empty lines, return False
        # This is a separate statement to make it more testable
        return self._handle_zero_non_empty_lines_for_count(non_empty_lines, matching_lines, has_explicit_header, threshold)

    def _handle_zero_non_empty_lines_for_count(self, count: int, matching_lines: int = 0, 
                                              has_explicit_header: bool = False, threshold: float = 0.5) -> bool:
        """
        Helper method to handle the case where there are zero non-empty lines.
        
        Args:
            count: Number of non-empty lines
            matching_lines: Number of lines matching TOC/index patterns
            has_explicit_header: Whether an explicit TOC/index header was found
            threshold: Threshold for determining if a page is TOC/index
            
        Returns:
            bool: True if the page is likely TOC or index, False otherwise
        """
        if count == 0:
            # Log and return False for zero non-empty lines
            logger.debug("No non-empty lines found in text, returning False")
            return False
        
        # For non-zero counts, continue with the normal processing
        match_ratio = matching_lines / count
        
        # Stricter threshold if no explicit header, slightly more lenient if header present
        required_threshold = threshold if has_explicit_header else threshold + 0.1
        
        # Also consider it TOC/Index if it has an explicit header and at least a few matching lines
        if has_explicit_header and match_ratio > 0.1:
            return True
            
        return match_ratio >= required_threshold

    def _log_zero_non_empty_lines(self):
        """Log a debug message when there are no non-empty lines."""
        logger.debug("No non-empty lines found in text, returning False")

    def _handle_zero_non_empty_lines(self) -> bool:
        """Helper method to handle the case where there are no non-empty lines."""
        return False

    def _handle_zero_non_empty_lines_with_logging(self) -> bool:
        """Helper method to handle the case where there are no non-empty lines, with logging."""
        self._log_zero_non_empty_lines()
        return self._handle_zero_non_empty_lines()

    def detect_toc_pages(self, pages_text: List[str]) -> List[int]:
        """
        Detect pages that appear to be table of contents pages.

        Args:
            pages_text: List of text content for each page

        Returns:
            List[int]: List of page indices (0-indexed) that are likely TOC pages
        """
        toc_pages_set = set()  # Use a set to avoid duplicates

        # Look for explicit TOC header on pages
        toc_header_patterns = [
            re.compile(r'^\s*Table\s+of\s+Contents\s*$', re.IGNORECASE),
            re.compile(r'^\s*Contents\s*$', re.IGNORECASE)
        ]

        initial_toc_pages = []
        # First pass: find pages with TOC headers
        for i, page_text in enumerate(pages_text):
            if not page_text:
                continue

            lines = page_text.strip().split('\n')
            for line in lines[:5]:  # Check first 5 lines for TOC header
                for pattern in toc_header_patterns:
                    if pattern.match(line.strip()):
                        initial_toc_pages.append(i)
                        break  # Found header on this page, move to next page
                if i in initial_toc_pages:  # Break outer loop if header found
                    break

        # Second pass: Check continuations and pages identified by content
        for i, page_text in enumerate(pages_text):
            # If it's an initial page or looks like TOC based on content
            if i in initial_toc_pages or self.is_toc_or_index_page(page_text, threshold=0.4):
                # Check if the *previous* page was already marked as TOC (indicates continuation)
                # Or if this page itself is strongly indicative
                if (i > 0 and i - 1 in toc_pages_set) or self.is_toc_or_index_page(page_text, threshold=0.5):
                    toc_pages_set.add(i)
                    # Check forward for continuations
                    j = i + 1
                    while j < len(pages_text) and self.is_toc_or_index_page(pages_text[j], threshold=0.4):
                        toc_pages_set.add(j)
                        j += 1

        # Add initial header pages even if they didn't meet content threshold alone
        for page_index in initial_toc_pages:
            toc_pages_set.add(page_index)

        return sorted(list(toc_pages_set))

    def detect_index_pages(self, pages_text: List[str]) -> List[int]:
        """
        Detect pages that appear to be index pages.

        Args:
            pages_text: List of text content for each page

        Returns:
            List[int]: List of page indices (0-indexed) that are likely index pages
        """
        index_pages_set = set()  # Use a set

        # Look for explicit index header on pages
        index_header_patterns = [
            re.compile(r'^\s*Index\s*$', re.IGNORECASE),
            re.compile(r'^\s*Subject\s+Index\s*$', re.IGNORECASE)
        ]

        # Patterns for index entries
        index_entry_patterns = [
            re.compile(r'^\s*[A-Za-z][\w\s,-]+,\s*\d+([\s,-]+\d+)*\s*$'),  # Term, 1, 5-10
            re.compile(r'^\s*[A-Za-z][\w\s]+[\. ]{3,}\s*\d+\s*$')  # Term......... 10
        ]

        initial_index_pages = []
        # Find pages with index headers, typically near the end of the document
        start_check_index = max(0, len(pages_text) - max(20, len(pages_text) // 3))  # Check last third or last 20 pages

        for i in range(start_check_index, len(pages_text)):
            page_text = pages_text[i]
            if not page_text:
                continue

            lines = page_text.strip().split('\n')
            for line in lines[:5]:  # Check first 5 lines for index header
                for pattern in index_header_patterns:
                    if pattern.match(line.strip()):
                        initial_index_pages.append(i)
                        break
                if i in initial_index_pages:
                    break

        # Add initial header pages
        for page_index in initial_index_pages:
            index_pages_set.add(page_index)

        # Second pass: Check continuations and pages identified by content
        for i in range(len(pages_text)):
            # If this is an initial index page or follows a known index page
            if i in index_pages_set or (i > 0 and i - 1 in index_pages_set):
                # If it's already marked as an index page, check the next page
                if i in index_pages_set:
                    # Check if the next page is a continuation
                    if i + 1 < len(pages_text):
                        next_page = pages_text[i + 1]
                        if next_page:
                            # Count index entry matches in the next page
                            lines = next_page.strip().split('\n')
                            entry_matches = 0
                            for line in lines:
                                for pattern in index_entry_patterns:
                                    if pattern.match(line.strip()):
                                        entry_matches += 1
                                        break

                            # If enough lines match index entry patterns, consider it a continuation
                            if entry_matches >= 2 or (entry_matches > 0 and entry_matches / len(lines) >= 0.2):
                                index_pages_set.add(i + 1)

                # If it's not already marked but follows an index page, check if it looks like an index
                elif self.is_toc_or_index_page(pages_text[i], threshold=0.4):
                    index_pages_set.add(i)

        return sorted(list(index_pages_set))

    def filter_line(self, line: str) -> bool:
        """
        Determine if a line should be kept or filtered out.

        Args:
            line: Text line to check

        Returns:
            bool: True if the line should be kept, False if it should be filtered out
        """
        stripped_line = line.strip()
        if not stripped_line:
            return False  # Skip empty lines

        # Check against all non-useful patterns
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.match(pattern, stripped_line):
                    logger.debug(f"Filtering out line due to {category} pattern: {stripped_line}")
                    return False  # Filter out this line

        return True  # Keep this line

    def filter_text(self, text: str) -> str:
        """
        Filter out non-RAG useful content from text.

        Args:
            text: Input text to filter

        Returns:
            str: Filtered text with non-RAG useful content removed
        """
        if not text:
            return ""

        lines = text.split('\n')
        filtered_lines = [line for line in lines if self.filter_line(line)]

        # Remove excessive blank lines that might result from filtering
        cleaned_text = re.sub(r'\n{3,}', '\n\n', '\n'.join(filtered_lines)).strip()

        return cleaned_text

    def filter_document_pages(
        self,
        pages_text: List[str],
        toc_pages: Optional[List[int]] = None,
        index_pages: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Filter out non-RAG useful content from document pages.

        Args:
            pages_text: List of text content for each page
            toc_pages: Known table of contents pages (if already detected)
            index_pages: Known index pages (if already detected)

        Returns:
            Tuple[List[str], List[int]]: Filtered pages text and list of excluded page indices
        """
        # Detect TOC and index pages if not provided
        detected_toc = toc_pages if toc_pages is not None else self.detect_toc_pages(pages_text)
        detected_index = index_pages if index_pages is not None else self.detect_index_pages(pages_text)

        # Combine pages to exclude
        exclude_pages = set(detected_toc + detected_index)
        logger.info(f"Excluding {len(exclude_pages)} pages as non-RAG useful content (TOC: {detected_toc}, Index: {detected_index})")

        # Filter remaining pages line by line
        filtered_pages = []
        for i, page_text in enumerate(pages_text):
            if i in exclude_pages:
                filtered_pages.append("")  # Empty string for excluded pages
            else:
                filtered_pages.append(self.filter_text(page_text))

        return filtered_pages, sorted(list(exclude_pages))

def filter_extracted_text(text_content: str) -> str:
    """
    Filter out non-RAG useful content from extracted text.

    This function removes headers, footers, page numbers, TOC/Index entries,
    and other non-informative content from document text.

    Args:
        text_content (str): The extracted text content to filter

    Returns:
        str: Filtered text with non-RAG useful content removed
    """
    if not text_content:
        return ""

    logger.debug("Filtering extracted text content")

    # Create a content filter with default English language
    content_filter = ContentFilter()

    # Apply filtering to the text
    filtered_text = content_filter.filter_text(text_content)

    # Log statistics about filtering
    original_length = len(text_content)
    filtered_length = len(filtered_text)
    reduction_percentage = ((original_length - filtered_length) / original_length) * 100 if original_length > 0 else 0

    logger.info(f"Text filtering complete. Original length: {original_length}, Filtered length: {filtered_length}, "
                f"Reduction: {reduction_percentage:.2f}%")

    return filtered_text
