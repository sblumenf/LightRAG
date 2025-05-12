"""
Diagram analysis module for LightRAG.

This module provides functionality to extract and analyze diagrams from PDF documents.
It uses PyMuPDF (fitz) to detect and extract images, and applies heuristics to
determine which images are likely diagrams.
"""
import logging
import os
import base64
import uuid
import io
from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np

# Try to import necessary libraries with graceful fallbacks
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Installing it is recommended for diagram extraction.")

try:
    import PIL
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Image processing capabilities will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Shape detection capabilities will be limited.")

# Import settings from config if available
try:
    from lightrag.config_loader import get_enhanced_config
    config = get_enhanced_config()
    DIAGRAM_DETECTION_THRESHOLD = config.diagram_detection_threshold
    DEFAULT_DIAGRAM_PROMPTS = {
        'general': "Analyze this diagram and provide a detailed textual description. Focus on identifying the diagram type, key components, and the relationships or processes it illustrates. Be concise and informative.",
        'flowchart': "Analyze this flowchart and describe the process and steps involved. Focus on the sequence of actions, decision points, and overall workflow. Be detailed and clear.",
        'bar_chart': "Analyze this bar chart and describe the data being presented, including comparisons and trends. Focus on categories, values, axes, and any significant patterns. Be analytical and precise.",
        'line_chart': "Analyze this line chart and describe the trends, patterns, and relationships shown. Focus on the variables, time periods, and significant changes. Be analytical and precise.",
        'pie_chart': "Analyze this pie chart and describe the proportions and relationships shown. Focus on the segments, percentages, and overall distribution. Be analytical and precise.",
        'scatter_plot': "Analyze this scatter plot and describe the relationship between variables. Focus on correlation, clusters, outliers, and trends. Be analytical and precise.",
        'network_diagram': "Analyze this network diagram and describe the nodes, connections, and overall structure. Focus on key components, relationships, and any central elements. Be detailed and clear.",
        'architecture_diagram': "Analyze this architecture diagram and describe the components, layers, and interactions. Focus on the system structure, data flow, and key interfaces. Be detailed and technical.",
        'uml_diagram': "Analyze this UML diagram and describe the classes, relationships, and overall structure. Focus on inheritance, associations, and key design patterns. Be detailed and technical.",
        'er_diagram': "Analyze this ER diagram and describe the entities, relationships, and attributes. Focus on cardinality, key fields, and overall data structure. Be detailed and technical."
    }
    LLM_PROVIDER = "custom"  # Default to custom since we'll use the LightRAG LLM
except ImportError:
    # Define fallback settings
    DIAGRAM_DETECTION_THRESHOLD = 0.6
    DEFAULT_DIAGRAM_PROMPTS = {'general': "Describe this diagram."}
    LLM_PROVIDER = "custom"
    logger.warning("Could not import config, using fallback defaults for diagram analysis.")


class DiagramAnalyzer:
    """
    Extract and analyze diagrams from documents, generating textual descriptions.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_service=None):
        """
        Initialize the diagram analyzer.

        Args:
            config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
            llm_service: Optional LLM service for diagram description
        """
        self.llm_service = llm_service
        
        # Load initial prompts and copy to avoid modification
        self.description_prompts = DEFAULT_DIAGRAM_PROMPTS.copy()
        
        # Default threshold
        self.diagram_detection_threshold = DIAGRAM_DETECTION_THRESHOLD

        # Override defaults from config if provided
        if config:
            self.diagram_detection_threshold = config.get('diagram_detection_threshold', self.diagram_detection_threshold)
            # Handle description_prompts override carefully
            config_prompts = config.get("description_prompts")
            if config_prompts:
                if not isinstance(config_prompts, dict):
                    logger.warning("description_prompts in config is not a dictionary, ignoring override.")
                else:
                    # Update the defaults with the provided custom prompts
                    self.description_prompts.update(config_prompts)
                    logger.info(f"Updated description prompts from config. Current keys: {list(self.description_prompts.keys())}")

    def add_description_template(self, diagram_type: str, prompt_template: str):
        """
        Adds or updates a description prompt template for a specific diagram type.

        Args:
            diagram_type (str): The type of diagram (e.g., 'flowchart', 'bar_chart').
            prompt_template (str): The prompt template string to use for this diagram type.
        """
        self.description_prompts[diagram_type] = prompt_template
        logger.info(f"Added/Updated description template for diagram type: {diagram_type}")

    def _create_description_prompt(self, diagram_type: str = 'general') -> str:
        """
        Creates the appropriate prompt for diagram description based on type.

        Args:
            diagram_type (str, optional): The type of diagram. Defaults to 'general'.

        Returns:
            str: The generated prompt string.
        """
        prompt_template = self.description_prompts.get(diagram_type, self.description_prompts.get('general', "Describe this diagram."))
        formatted_diagram_type = diagram_type.replace('_', ' ').title()
        try:
            final_prompt = prompt_template.format(diagram_type=formatted_diagram_type)
        except KeyError:
            final_prompt = prompt_template
        keywords = ["description", "analyze", "explain", "interpret", "summarize"]
        if not any(keyword in final_prompt.lower() for keyword in keywords):
             final_prompt += "\n\nPlease provide a detailed textual description of this diagram."
        logger.debug(f"Using prompt for diagram type '{diagram_type}': {final_prompt[:150]}...")
        return final_prompt

    def _calculate_diagram_score(self, image_data: bytes, image_info: Dict[str, Any]) -> float:
        """
        Calculates a score indicating the likelihood of an image being a diagram.

        Args:
            image_data (bytes): Image data of the diagram.
            image_info (Dict[str, Any]): Metadata about the image.

        Returns:
            float: A score indicating likelihood.
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available for diagram scoring. Returning default score 1.0.")
            return 1.0

        try:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.warning(f"Error opening image for diagram scoring: {e}")
            return 0.0

        width, height = img.size
        if width == 0 or height == 0:
            logger.warning("Image has zero width or height, cannot score.")
            return 0.0

        aspect_ratio = width / height
        color_ratio = self._calculate_color_ratio(img)
        edge_density = self._calculate_edge_density(img)
        shape_score = 0.0

        if CV2_AVAILABLE:
            try:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                thresh = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                shape_count = 0
                min_contour_area = 50
                for contour in contours:
                    if cv2.contourArea(contour) < min_contour_area: continue
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0: continue
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    if len(approx) >= 3: shape_count += 1
                shape_score = min(shape_count / 50.0, 0.3)
                logger.debug(f"Detected {shape_count} significant shapes. Shape score: {shape_score:.2f}")
            except Exception as e:
                logger.warning(f"Error during shape detection for scoring: {e}")
                shape_score = 0.0
        else: 
            shape_score = 0.0

        diagram_score = 0.0
        if 0.8 < aspect_ratio < 1.2: diagram_score -= 0.1
        elif aspect_ratio < 0.5: diagram_score -= 0.2
        elif aspect_ratio > 2.0: diagram_score += 0.4
        if color_ratio < 0.15: diagram_score += 0.3
        elif color_ratio > 0.8: diagram_score -= 0.2
        if edge_density > 0.15: diagram_score += 0.3
        elif edge_density < 0.02: diagram_score -= 0.1
        diagram_score += shape_score
        logger.debug(f"Calculated diagram score: {diagram_score:.2f} (AR: {aspect_ratio:.2f}, CR: {color_ratio:.2f}, ED: {edge_density:.2f}, SS: {shape_score:.2f})")
        return diagram_score

    def _is_diagram(self, image_data: bytes, image_info: Dict[str, Any]) -> bool:
        """
        Determine if an image is likely a diagram based on heuristics and score threshold.

        Args:
            image_data (bytes): Image data of the diagram.
            image_info (Dict[str, Any]): Metadata about the image.

        Returns:
            bool: True if likely a diagram, False otherwise.
        """
        diagram_score = self._calculate_diagram_score(image_data, image_info)
        is_diagram = diagram_score >= self.diagram_detection_threshold
        logger.debug(f"Checking diagram status: Score={diagram_score:.2f}, Threshold={self.diagram_detection_threshold}, IsDiagram={is_diagram}")
        return is_diagram

    def _calculate_color_ratio(self, img: Image.Image) -> float:
        """Calculates the ratio of non-white pixels to total pixels."""
        try:
            if img.mode == 'RGBA': img = img.convert('RGB')
            total_pixels = img.width * img.height
            if total_pixels == 0: return 1.0
            max_pixels_for_getcolors = 1_000_000
            if total_pixels <= max_pixels_for_getcolors:
                 colors = img.getcolors(total_pixels)
                 if colors: return min(len(colors) / 256.0, 1.0)
                 else: logger.debug("getcolors() returned None, falling back to sampling.")
            logger.debug("Using pixel sampling for color ratio calculation.")
            sample_size = 100; distinct_colors = set()
            for x in range(0, img.width, max(1, img.width // sample_size)):
                for y in range(0, img.height, max(1, img.height // sample_size)):
                    distinct_colors.add(img.getpixel((x, y)))
            return min(len(distinct_colors) / 256.0, 1.0)
        except Exception as e:
            logger.warning(f"Error calculating color ratio: {e}"); return 1.0

    def _calculate_edge_density(self, img: Image.Image) -> float:
        """Calculates the edge density of an image using edge detection."""
        try:
            edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
            edge_pixels = 0; threshold = 50
            for pixel_value in edges.getdata():
                if pixel_value > threshold: edge_pixels += 1
            total_pixels = img.width * img.height
            return edge_pixels / total_pixels if total_pixels else 0.0
        except Exception as e:
            logger.warning(f"Error calculating edge density: {e}"); return 0.0

    def extract_diagrams_from_pdf(self, pdf_path: str, diagram_type: str = 'general') -> List[Dict[str, Any]]:
        """
        Extract diagrams from a PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            diagram_type (str, optional): Default diagram type. Defaults to 'general'.
            
        Returns:
            List[Dict[str, Any]]: List of extracted diagrams with metadata
        """
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available. Cannot extract diagrams.")
            return []
            
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
            
        extracted_diagrams = []
        min_size = 100  # Minimum size for images to be considered diagrams
        
        try:
            with fitz.open(pdf_path) as pdf:
                logger.info(f"Processing PDF for diagrams: {pdf_path} ({len(pdf)} pages)")
                
                for page_index, page in enumerate(pdf):
                    page_num = page_index + 1
                    page_width, page_height = page.rect.width, page.rect.height
                    logger.debug(f"Processing Page {page_num}/{len(pdf)}")
                    
                    # Get all images on the page
                    image_list = page.get_images(full=True)
                    logger.debug(f"Page {page_num}: Found {len(image_list)} raw image objects.")
                    
                    processed_xrefs = set()  # Track processed image references
                    
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        if xref in processed_xrefs:
                            continue
                            
                        try:
                            # Extract the image
                            base_image = pdf.extract_image(xref)
                            if not base_image or not base_image.get("image"):
                                logger.warning(f"P{page_num},I{img_index}({xref}): Failed to extract image")
                                continue
                                
                            image_data = base_image["image"]
                            image_ext = base_image["ext"]
                            image_format = base_image.get("colorspace", "unknown")
                            
                            # Get image dimensions
                            width, height = 0, 0
                            if PIL_AVAILABLE:
                                try:
                                    img_pil = Image.open(io.BytesIO(image_data))
                                    width, height = img_pil.size
                                except Exception as pil_e:
                                    logger.warning(f"P{page_num},I{img_index}({xref}): PIL error: {pil_e}. Using PDF dimensions.")
                                    width, height = base_image.get("width", 0), base_image.get("height", 0)
                            else:
                                width, height = base_image.get("width", 0), base_image.get("height", 0)
                                
                            logger.debug(f"P{page_num},I{img_index}({xref}): Extracted {image_ext}, Size {width}x{height}")
                            
                            # Skip small images
                            if width < min_size or height < min_size:
                                logger.debug(f"Skipping small image ({width}x{height}).")
                                continue
                                
                            # Check if the image is a diagram
                            is_diagram = self._is_diagram(image_data, {
                                'page_width': page_width, 
                                'page_height': page_height, 
                                'format': image_format
                            })
                            
                            if is_diagram:
                                logger.info(f"P{page_num},I{img_index}({xref}): Identified as diagram.")
                                
                                # Generate a unique ID for the diagram
                                img_id = f"diagram-{uuid.uuid4()}"
                                
                                # Convert image data to base64
                                image_b64 = base64.b64encode(image_data).decode('utf-8')
                                
                                # Try to get the image position on the page
                                image_rect_list = None
                                try:
                                    bbox = page.get_image_bbox(img_info, transform=False)
                                    if bbox:
                                        # Convert fitz.Rect to list [x0, y0, x1, y1]
                                        image_rect_list = [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                                        logger.debug(f"P{page_num},I{img_index}({xref}): Found bbox: {image_rect_list}")
                                    else:
                                        logger.warning(f"P{page_num},I{img_index}({xref}): No bbox from get_image_bbox.")
                                except Exception as rect_e:
                                    logger.warning(f"P{page_num},I{img_index}({xref}): Error getting bbox: {rect_e}")
                                
                                # Try to get surrounding text for context
                                surrounding_text = ""
                                if image_rect_list:
                                    try:
                                        # Expand the rectangle slightly to capture nearby text
                                        expansion = 50
                                        text_rect = fitz.Rect(
                                            max(0, image_rect_list[0] - expansion),
                                            max(0, image_rect_list[1] - expansion),
                                            min(page_width, image_rect_list[2] + expansion),
                                            min(page_height, image_rect_list[3] + expansion)
                                        )
                                        surrounding_text = page.get_text("text", clip=text_rect).strip()
                                        logger.debug(f"P{page_num},I{img_index}({xref}): Text len {len(surrounding_text)}.")
                                    except Exception as text_e:
                                        logger.warning(f"P{page_num},I{img_index}({xref}): Error getting text: {text_e}")
                                
                                # Create diagram entry
                                diagram_entry = {
                                    'diagram_id': img_id,
                                    'page': page_num,
                                    'width': width,
                                    'height': height,
                                    'format': image_ext,
                                    'is_diagram': True,
                                    'position': image_rect_list,
                                    'surrounding_text': surrounding_text,
                                    'caption': None,
                                    'file_path': None,
                                    '_full_base64': image_b64,
                                    'base64_data': image_b64[:100] + '...',  # Truncated for logging
                                    'description': None,
                                    'extraction_method': 'pymupdf_image',
                                    'diagram_type': diagram_type
                                }
                                
                                extracted_diagrams.append(diagram_entry)
                                processed_xrefs.add(xref)
                            else:
                                logger.debug(f"P{page_num},I{img_index}({xref}): Skipped non-diagram.")
                                
                        except Exception as img_proc_e:
                            logger.warning(f"P{page_num}: Error processing img xref {xref}: {img_proc_e}")
                    
                    # Look for captions
                    try:
                        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
                        for block in blocks:
                            if block["type"] == 0:  # Text block
                                block_text = "".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
                                
                                # Look for figure/table captions
                                caption_match = re.match(r'(Figure|Fig\.?|Table)\s+(\d+[\.\d]*)\s*[:.-]?\s*(.*)', block_text, re.IGNORECASE)
                                if caption_match:
                                    caption_text = block_text
                                    caption_rect = fitz.Rect(block["bbox"])
                                    logger.debug(f"P{page_num}: Potential caption: '{caption_text[:50]}...'")
                                    
                                    # Find the closest diagram to this caption
                                    closest_diagram = None
                                    min_distance = float('inf')
                                    
                                    for diagram in extracted_diagrams:
                                        if diagram['page'] == page_num and diagram['position']:
                                            img_rect = fitz.Rect(diagram['position'])
                                            
                                            # Calculate vertical distance and horizontal overlap
                                            vertical_dist = min(
                                                abs(caption_rect.y0 - img_rect.y1),
                                                abs(caption_rect.y1 - img_rect.y0)
                                            )
                                            h_overlap = max(0, min(caption_rect.x1, img_rect.x1) - max(caption_rect.x0, img_rect.x0))
                                            
                                            # Consider captions close to the diagram with some horizontal overlap
                                            if vertical_dist < 100 and h_overlap > 0:
                                                distance = vertical_dist + (img_rect.width - h_overlap) * 0.1
                                                if distance < min_distance:
                                                    min_distance = distance
                                                    closest_diagram = diagram
                                    
                                    if closest_diagram:
                                        logger.info(f"P{page_num}: Associated caption '{caption_text[:50]}...' with diagram {closest_diagram['diagram_id']}")
                                        closest_diagram['caption'] = caption_text
                                    else:
                                        logger.debug(f"P{page_num}: Caption '{caption_text[:50]}...' has no matching diagram.")
                    except Exception as caption_e:
                        logger.warning(f"P{page_num}: Error processing captions: {caption_e}")
                
                logger.info(f"Finished PDF. Extracted {len(extracted_diagrams)} diagrams.")
                return extracted_diagrams
                
        except Exception as e:
            logger.error(f"Fatal error extracting diagrams from PDF '{pdf_path}': {str(e)}")
            return []

    def generate_diagram_description(self, diagram_data: Dict[str, Any], diagram_type: str = 'general') -> str:
        """
        Generates a textual description for a diagram.
        
        Args:
            diagram_data (Dict[str, Any]): Diagram data including base64 image
            diagram_type (str, optional): Type of diagram. Defaults to 'general'.
            
        Returns:
            str: Textual description of the diagram
        """
        # Get the base64 image data
        image_data = diagram_data.get('_full_base64')
        surrounding_text = diagram_data.get('surrounding_text', '')
        caption = diagram_data.get('caption', '')
        
        if not image_data:
            logger.error("Missing '_full_base64' in diagram data.")
            return "Error: Missing image data."
            
        # Create the prompt for description
        prompt = self._create_description_prompt(diagram_type=diagram_type)
        diagram_data['diagram_type'] = diagram_type
        logger.debug(f"Generating description for diagram {diagram_data.get('diagram_id', 'N/A')} type: {diagram_type}")
        
        # Add context to the prompt if available
        context = ""
        if caption:
            context += f"Caption: \"{caption}\"\n\n"
        if surrounding_text and (not caption or caption not in surrounding_text):
            context += f"Context: {surrounding_text}\n\n"
            
        full_prompt = f"{context}{prompt}"
        
        # If we have an LLM service, use it to generate a description
        if self.llm_service:
            try:
                # Decode the base64 image
                img_bytes = base64.b64decode(image_data)
                
                # Call the LLM service with the image and prompt
                # This is a placeholder - actual implementation will depend on the LLM service interface
                response = "This is a placeholder diagram description. In a real implementation, this would be generated by an LLM."
                
                return response
            except Exception as e:
                logger.error(f"Error generating diagram description: {str(e)}")
                return f"Error generating description: {str(e)}"
        else:
            # If no LLM service is available, return a basic description
            basic_desc = f"Diagram on page {diagram_data.get('page', 'unknown')}"
            if caption:
                basic_desc += f" with caption: {caption}"
            if diagram_data.get('width') and diagram_data.get('height'):
                basic_desc += f". Dimensions: {diagram_data.get('width')}x{diagram_data.get('height')} pixels"
            
            return basic_desc

# Add the missing detect_diagrams function that's being imported by tests
def detect_diagrams(pdf_path: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Detect diagrams in a PDF document.
    
    This function is a wrapper around DiagramAnalyzer for backward compatibility.
    
    Args:
        pdf_path (str): Path to the PDF file
        **kwargs: Additional arguments for diagram detection
        
    Returns:
        List[Dict[str, Any]]: List of detected diagrams with metadata
    """
    analyzer = DiagramAnalyzer()
    return analyzer.extract_diagrams_from_pdf(pdf_path, **kwargs)