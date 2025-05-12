"""
Tests targeting specific edge cases in diagram analyzer that may be encountered
during future functionality extensions.

This module focuses on practical, targeted tests for error paths in:
1. PDF extraction and processing
2. External API integrations
3. Image processing error handling

These tests are designed to provide coverage for the most likely error scenarios
that would affect new functionality built on the diagram analyzer.
"""

import os
import io
import base64
import tempfile
import unittest
import json
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

import pytest
from PIL import Image, ImageFilter

from document_processing.diagram_analyzer import DiagramAnalyzer


class TestPDFExtractionEdgeCases(unittest.TestCase):
    """Tests for PDF extraction error paths that may affect new functionality."""
    
    def setUp(self):
        self.config = {'diagram_detection_threshold': 0.6}
        self.analyzer = DiagramAnalyzer(config=self.config)
        
    @patch('document_processing.diagram_analyzer.fitz')
    def test_pdf_page_extraction_errors(self, mock_fitz):
        """Test handling of errors during PDF page processing."""
        # Create a mock PDF document
        mock_doc = MagicMock()
        
        # First page raises an exception during processing
        bad_page = MagicMock()
        bad_page.get_images.side_effect = Exception("Corrupt page data")
        
        # Second page works fine
        good_page = MagicMock()
        good_page.get_images.return_value = []
        good_page.rect = MagicMock()
        good_page.rect.width, good_page.rect.height = 800, 1000
        
        # Set up the document with both pages
        mock_doc.__enter__.return_value = [bad_page, good_page]
        mock_fitz.open.return_value = mock_doc
        mock_fitz.TEXTFLAGS_TEXT = 0
        
        # Test extraction with a partially corrupted PDF
        result = self.analyzer.extract_diagrams_from_pdf('/fake/path/to/corrupt.pdf')
        
        # We should still get processing of the good page
        self.assertEqual(len(result), 0)  # No images on the good page
        
    def test_empty_pdf_path(self):
        """Test handling of empty PDF path."""
        result = self.analyzer.extract_diagrams_from_pdf('')
        self.assertEqual(result, [])
        
    @patch('document_processing.diagram_analyzer.os.path.exists')
    @patch('document_processing.diagram_analyzer.fitz')
    def test_pdf_with_unicode_path(self, mock_fitz, mock_exists):
        """Test handling of PDFs with Unicode characters in the path."""
        # Setup
        mock_exists.return_value = True
        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = []
        mock_fitz.open.return_value = mock_doc
        
        # Test with Unicode path
        unicode_path = '/path/to/документ-with-unicode-characters.pdf'
        result = self.analyzer.extract_diagrams_from_pdf(unicode_path)
        
        # Should handle the path correctly
        self.assertEqual(result, [])
        mock_fitz.open.assert_called_once_with(unicode_path)


class TestAPIIntegrationEdgeCases(unittest.TestCase):
    """Tests for API integration error paths that may affect new functionality."""
    
    def setUp(self):
        self.config = {
            'diagram_detection_threshold': 0.6,
            'enable_diagram_description_cache': True,
            'diagram_description_cache_dir': tempfile.mkdtemp(),
        }
        self.analyzer = DiagramAnalyzer(config=self.config)
        
        # Create test diagram data
        img = Image.new('RGB', (100, 100), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = buffer.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        self.test_diagram = {
            'diagram_id': 'test-api-001',
            'page': 1,
            '_full_base64': base64_image,
            'width': 100,
            'height': 100
        }
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self):
        """Test handling of API timeouts in vision adapter."""
        # Setup a mock adapter that times out
        mock_adapter = AsyncMock()
        mock_adapter.provider_name = "test"
        timeout_error = TimeoutError("API request timed out")
        mock_adapter.generate_description.side_effect = timeout_error
        
        # Set the mock adapter
        self.analyzer.vision_adapter = mock_adapter
        
        # Setup a mock LLM service for fallback
        mock_llm = AsyncMock()
        mock_llm.generate_text = AsyncMock(return_value="Fallback description after timeout")
        self.analyzer.llm_service = mock_llm
        
        # Test with timeout
        result = await self.analyzer.generate_diagram_description(self.test_diagram)
        
        # Should use fallback
        self.assertEqual(result, "Fallback description after timeout")
        mock_adapter.generate_description.assert_called_once()
        mock_llm.generate_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self):
        """Test handling of API rate limits."""
        # Setup a mock adapter that gets rate limited
        mock_adapter = AsyncMock()
        mock_adapter.provider_name = "test"
        rate_limit_error = Exception("Rate limit exceeded")
        mock_adapter.generate_description.side_effect = rate_limit_error
        
        # Set the mock adapter
        self.analyzer.vision_adapter = mock_adapter
        
        # Setup a mock LLM service for fallback
        mock_llm = AsyncMock()
        mock_llm.generate_text = AsyncMock(return_value="Fallback description after rate limit")
        self.analyzer.llm_service = mock_llm
        
        # Test with rate limit
        result = await self.analyzer.generate_diagram_description(self.test_diagram)
        
        # Should use fallback
        self.assertEqual(result, "Fallback description after rate limit")
        mock_adapter.generate_description.assert_called_once()
        mock_llm.generate_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_connection_error_handling(self):
        """Test handling of connection errors."""
        # Setup a mock adapter with connection error
        mock_adapter = AsyncMock()
        mock_adapter.provider_name = "test"
        connection_error = ConnectionError("Failed to connect to API")
        mock_adapter.generate_description.side_effect = connection_error
        
        # Set the mock adapter
        self.analyzer.vision_adapter = mock_adapter
        
        # Setup a mock LLM service for fallback that also fails
        mock_llm = AsyncMock()
        mock_llm.generate_text = AsyncMock(side_effect=ConnectionError("Failed to connect to LLM"))
        self.analyzer.llm_service = mock_llm
        
        # Test with connection errors for both primary and fallback
        result = await self.analyzer.generate_diagram_description(self.test_diagram)
        
        # Should return basic diagram info when all else fails
        self.assertIn("Diagram on page 1", result)
        self.assertIn("Dimensions: 100x100", result)
        mock_adapter.generate_description.assert_called_once()
        mock_llm.generate_text.assert_called_once()


class TestImageProcessingEdgeCases(unittest.TestCase):
    """Tests for image processing error paths that may affect new functionality."""
    
    def setUp(self):
        self.config = {'diagram_detection_threshold': 0.6}
        self.analyzer = DiagramAnalyzer(config=self.config)
    
    def test_corrupt_image_data_handling(self):
        """Test handling of corrupt image data."""
        # Create corrupt image data
        corrupt_data = b'not-a-valid-image-file'
        
        # Test with corrupt data
        score = self.analyzer._calculate_diagram_score(corrupt_data, {})
        
        # Should return a score of 0 for corrupt data
        self.assertEqual(score, 0.0)
    
    @patch('PIL.Image.open')
    def test_image_with_alpha_channel(self, mock_open):
        """Test handling of images with alpha channel."""
        # Create a test image with alpha
        img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 128))
        mock_open.return_value = img
        
        # Test with RGBA image
        score = self.analyzer._calculate_diagram_score(b'mock-image-data', {})
        
        # Should process the image and return a valid score
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)
    
    @patch('PIL.Image.open')
    def test_tiny_image_handling(self, mock_open):
        """Test handling of very small images."""
        # Create a tiny image
        img = Image.new('RGB', (10, 10), color='white')
        mock_open.return_value = img
        
        # Test with tiny image
        score = self.analyzer._calculate_diagram_score(b'mock-tiny-image', {})
        
        # Should process the image and return a valid score
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)
    
    @patch('PIL.Image.open')
    def test_unusual_color_modes(self, mock_open):
        """Test handling of images with unusual color modes."""
        # Create a test image with grayscale mode
        img = Image.new('L', (100, 100), color=128)  # 'L' mode is 8-bit grayscale
        mock_open.return_value = img
        
        # Test with grayscale image
        score = self.analyzer._calculate_diagram_score(b'mock-grayscale-image', {})
        
        # Should process the image and return a valid score
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == '__main__':
    unittest.main()