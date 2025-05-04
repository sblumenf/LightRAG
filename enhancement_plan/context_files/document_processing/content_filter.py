"""
Content filtering module for removing non-RAG useful elements from documents.
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
                r'^\s*Page \d+(?: of \d+)?\s*$',  # Page X or Page X of Y (Improved)
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
                # --- Start Added/Modified Patterns ---
                # Matches lines like "Chapter 1: Title....... 5" or "1.2 Section...10"
                r'^\s*(?:Chapter\s+[\dIVXLCDMivxlcdm]+|[\d\.]+|Appendix\s+[A-Z])[:\.\s].*?[\. ]{3,}\s*\d+\s*$',
                # General pattern for lines ending with dots/spaces and page number
                r'^\s*.*[\. ]{3,}\s*\d+\s*$',
                # --- End Added/Modified Patterns ---
            ],

            # Index patterns
            'index': [
                r'^\s*Index\s*$',
                r'^\s*INDEX\s*$',
                r'^\s*Subject\s+Index\s*$',
                # --- Start Added/Modified Patterns ---
                # Matches lines like "Term, 1, 5-10" or "Another Term, 23"
                r'^\s*[A-Za-z][\w\s,-]+,\s*\d+([\s,-]+\d+)*\s*$',
                 # General pattern for lines ending with dots/spaces and page number (can also appear in indices)
                r'^\s*.*[\. ]{3,}\s*\d+\s*$',
                # --- End Added/Modified Patterns ---
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

        # Additional patterns for English language specifics (already included in main TOC pattern)
        # if self.language == 'en':
        #     self.patterns['toc'].extend([
        #         r'^\s*Chapter\s+[IVXLCDMivxlcdm]+[\.:]?\s+.*\s+\d+\s*$',  # Roman numeral chapters
        #     ]) # This is now covered by the more general TOC pattern above

    def is_toc_or_index_page(self, text: str, threshold: float = 0.5) -> bool: # Lowered threshold slightly
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
        compiled_patterns = [re.compile(p) for p in toc_index_patterns] # Compile for efficiency

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
                    break # Move to next line once a match is found

        if non_empty_lines == 0:
            return False

        match_ratio = matching_lines / non_empty_lines

        # Stricter threshold if no explicit header, slightly more lenient if header present
        required_threshold = threshold if has_explicit_header else threshold + 0.1

        # Also consider it TOC/Index if it has an explicit header and at least a few matching lines
        if has_explicit_header and match_ratio > 0.1:
             return True

        return match_ratio >= required_threshold

    def detect_toc_pages(self, pages_text: List[str]) -> List[int]:
        """
        Detect pages that appear to be table of contents pages.

        Args:
            pages_text: List of text content for each page

        Returns:
            List[int]: List of page indices (0-indexed) that are likely TOC pages
        """
        toc_pages_set = set() # Use a set to avoid duplicates

        # Look for explicit TOC header on pages
        toc_header_patterns = [
            re.compile(r'^\s*Table\s+of\s+Contents\s*$', re.IGNORECASE), # Added IGNORECASE
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
                        break # Found header on this page, move to next page
                if i in initial_toc_pages: # Break outer loop if header found
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
        index_pages_set = set() # Use a set

        # Look for explicit index header on pages
        index_header_patterns = [
            re.compile(r'^\s*Index\s*$', re.IGNORECASE),
            re.compile(r'^\s*Subject\s+Index\s*$', re.IGNORECASE)
        ]

        initial_index_pages = []
        # Find pages with index headers, typically near the end of the document
        start_check_index = max(0, len(pages_text) - max(20, len(pages_text) // 3)) # Check last third or last 20 pages

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

        # Second pass: Check continuations and pages identified by content
        for i in range(start_check_index, len(pages_text)):
             page_text = pages_text[i]
             # If it's an initial page or looks like Index based on content
             if i in initial_index_pages or self.is_toc_or_index_page(page_text, threshold=0.4):
                  # Check if the *previous* page was already marked as Index (indicates continuation)
                  # Or if this page itself is strongly indicative
                  if (i > 0 and i - 1 in index_pages_set) or self.is_toc_or_index_page(page_text, threshold=0.5):
                       index_pages_set.add(i)
                       # Check forward for continuations
                       j = i + 1
                       # Check only a few pages ahead for index continuation, less likely to span many pages
                       while j < min(len(pages_text), i + 4) and self.is_toc_or_index_page(pages_text[j], threshold=0.4):
                            index_pages_set.add(j)
                            j += 1

        # Add initial header pages even if they didn't meet content threshold alone
        for page_index in initial_index_pages:
            index_pages_set.add(page_index)

        return sorted(list(index_pages_set))


    def filter_line(self, line: str) -> bool:
        """
        Determine if a line should be filtered out.

        Args:
            line: Text line to check

        Returns:
            bool: True if the line should be kept, False if it should be filtered out
        """
        stripped_line = line.strip()
        if not stripped_line:
            return False  # Skip empty lines

        # Check against all non-useful patterns
        # Compile patterns once if this method is called very frequently,
        # but for typical document processing, recompiling here is acceptable.
        for category, patterns in self.patterns.items():
            # Prioritize TOC/Index/Header/Footer checks as they are common filters
            if category in ['toc', 'index', 'header_footer', 'references', 'disclaimers']:
                for pattern in patterns:
                    if re.match(pattern, stripped_line):
                        # print(f"Filtering line due to pattern {pattern}: {stripped_line}") # Debugging
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

    def extract_formulas(self, text: str) -> List[Dict[str, Union[str, Tuple[int, int]]]]:
        """
        Extract mathematical formulas from text.
        (Keeping this method as is, assuming it passed tests)

        Args:
            text: Input text to search for formulas

        Returns:
            List[Dict[str, str]]: List of extracted formulas with their context
        """
        formulas = []

        # Simple formula detection patterns
        formula_patterns = [
            # Basic equation pattern with equal sign
            r'([A-Za-z0-9_\s\+\-\*\/\(\)\[\]\{\}=><≤≥±∑∏∫√∛∂∞≈≠∝∅∈∉∩∪⊂⊃⊄⊅⊆⊇⊕⊗ΛαβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]+)\s*=\s*([\w\s\+\-\*\/\(\)\[\]\{\}=><≤≥±∑∏∫√∛∂∞≈≠∝∅∈∉∩∪⊂⊃⊄⊅⊆⊇⊕⊗ΛαβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]+)', # Added Greek letters

            # Mathematical expressions with common symbols
            r'([A-Za-z0-9_]+\s*[\+\-\*\/]\s*[A-Za-z0-9_]+(?:\s*[\+\-\*\/]\s*[A-Za-z0-9_]+)*)', # Allow longer chains

            # Expressions with superscripts/subscripts represented textually (basic)
            r'([A-Za-z0-9_]+(?:[\^_]\{?[\d\w\+\-]+\}?)+)', # Match x^2, y_1, z^{n+1}

            # LaTeX-like formulas
            r'(\$[^\$]+\$)',
            r'(\\\([^\\]+\\\))', # Corrected escaping for \( ... \)
            r'(\\\[[^\\]+\\\])', # Corrected escaping for \[ ... \]
        ]

        # Extract formulas and surrounding context
        extracted_spans = set() # Avoid overlapping matches from different patterns
        context_chars = 75 # Reduced context slightly

        for pattern in formula_patterns:
            try:
                compiled_pattern = re.compile(pattern)
                matches = compiled_pattern.finditer(text)
                for match in matches:
                    span = match.span()
                    # Check if this span overlaps significantly with an already extracted one
                    is_overlapping = False
                    for ext_span in extracted_spans:
                        if max(span[0], ext_span[0]) < min(span[1], ext_span[1]):
                            # If the new match is much smaller and inside the old one, skip it
                            if span[0] >= ext_span[0] and span[1] <= ext_span[1] and (span[1]-span[0]) < (ext_span[1]-ext_span[0]) * 0.8:
                                is_overlapping = True
                                break
                            # If the old one is much smaller and inside the new one, remove the old one
                            if ext_span[0] >= span[0] and ext_span[1] <= span[1] and (ext_span[1]-ext_span[0]) < (span[1]-span[0]) * 0.8:
                                extracted_spans.remove(ext_span)
                                break # Check against others continues

                    if is_overlapping:
                        continue

                    formula_text = match.group(0).strip()
                    if len(formula_text) < 3: # Skip very short matches
                         continue

                    # Get surrounding context
                    start = max(0, match.start() - context_chars)
                    end = min(len(text), match.end() + context_chars)

                    context_before = text[start:match.start()].strip()
                    # Remove trailing parts of lines from context_before
                    context_before = context_before.split('\n')[-1]

                    context_after = text[match.end():end].strip()
                    # Remove leading parts of lines from context_after
                    context_after = context_after.split('\n')[0]

                    formulas.append({
                        'formula': formula_text,
                        'context_before': context_before,
                        'context_after': context_after,
                        'position': span
                    })
                    extracted_spans.add(span)
            except re.error as e:
                 logger.warning(f"Regex error in formula extraction pattern '{pattern}': {e}")


        # Sort formulas by position
        formulas.sort(key=lambda x: x['position'][0])

        return formulas

