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
import json
import time
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
import re
import numpy as np
from pathlib import Path

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
        # General diagram prompt
        'general': """
You are analyzing a diagram extracted from an educational document that will be used for exam preparation.
Provide a comprehensive, educational description of this diagram following these guidelines:

1. Start by identifying what type of diagram this is (flowchart, architecture diagram, organizational chart, etc.)
2. Describe the main components or elements shown in the diagram
3. Explain the relationships, connections, or processes illustrated
4. Highlight the key concept or insight that this diagram is intended to communicate
5. If there are any labels, titles, or text visible in the diagram, incorporate this information
6. Keep the explanation educational, technical, and precise
7. Focus on information that would be relevant for a student preparing for an exam

Your description should be well-structured, educational, and thorough without unnecessary detail.
""",

        # Flow-related diagrams
        'flowchart': """
This is a flowchart from an educational document. Provide a comprehensive explanation that:

1. Identifies the process or workflow being depicted
2. Describes each step in the process sequentially
3. Explains decision points, branches, and alternate paths
4. Notes any loops or repeating segments
5. Identifies start and end points
6. Explains what happens at each step and transition
7. Summarizes the overall purpose of this process flow
8. Highlights any special conditions or exception handling
9. Includes numerical values or time periods if present

Focus on educational clarity and precision for students preparing for exams.
""",

        'process_diagram': """
This process diagram appears in educational material. Provide a detailed explanation that:

1. Identifies the overall process or system being depicted
2. Describes each stage or phase in sequence
3. Explains the inputs and outputs at each stage
4. Describes how components interact or transform
5. Highlights key decision points or critical steps
6. Explains feedback loops or cyclical elements if present
7. Connects this process to its broader context or purpose
8. Notes any constraints, requirements, or conditions
9. Includes quantitative information if present (rates, times, etc.)

Your description should be educational, technical, and thorough for students preparing for exams.
""",

        # Data visualization diagrams
        'bar_chart': """
This bar chart appears in educational material. Analyze it thoroughly with:

1. Identification of what is being measured or compared
2. Description of the axes (x-axis and y-axis) and their units
3. Analysis of the highest and lowest values
4. Identification of significant patterns, trends, or outliers
5. Comparison between different categories or groups
6. Quantitative assessment with specific values when visible
7. Context for what these data represent in the subject matter
8. Potential implications or conclusions from the data
9. Any limitations or special considerations visible in the chart

Your analysis should be educational, data-focused, and provide insights a student would need for exam preparation.
""",

        'line_chart': """
This line chart appears in educational material. Analyze it thoroughly with:

1. Identification of what trends or changes are being tracked
2. Description of the axes (x-axis and y-axis) and their units
3. Analysis of major trend directions (increasing, decreasing, fluctuating)
4. Identification of significant points, peaks, valleys, or inflection points
5. Comparison between different lines if multiple data series are present
6. Quantitative assessment with specific values at key points
7. Context for what these trends represent in the subject matter
8. Potential implications or conclusions from the trends
9. Any limitations or special considerations visible in the chart

Your analysis should be educational, trend-focused, and provide insights a student would need for exam preparation.
""",

        'pie_chart': """
This pie chart appears in educational material. Analyze it thoroughly with:

1. Identification of what is being categorized or distributed
2. Description of the segments from largest to smallest
3. Specific percentages or proportions for each segment when visible
4. Analysis of the relative sizes and their significance
5. Identification of the dominant or majority categories
6. Context for what this distribution represents in the subject matter
7. Comparison between segments that are notably similar or different
8. Potential implications or conclusions from this distribution
9. Any limitations or special considerations visible in the chart

Your analysis should be educational, focused on proportional relationships, and provide insights a student would need for exam preparation.
""",

        'scatter_plot': """
This scatter plot appears in educational material. Analyze it thoroughly with:

1. Identification of the variables on each axis and their units
2. Description of the overall pattern of data points
3. Analysis of correlation (positive, negative, or none)
4. Identification of clusters or groupings
5. Description of outliers or unusual data points
6. Assessment of the strength of relationship (strong, moderate, weak)
7. Context for what this relationship represents in the subject matter
8. Potential implications or cause-effect relationships
9. Any limitations or special considerations visible in the plot

Your analysis should be educational, relationship-focused, and provide insights a student would need for exam preparation.
""",

        # Technical diagrams
        'network_diagram': """
This network diagram appears in educational material. Provide a detailed explanation that:

1. Identifies the type of network being depicted (computer network, social network, etc.)
2. Describes the nodes/entities and what they represent
3. Explains the connections/edges and their significance
4. Identifies central or highly connected nodes
5. Describes any clusters, groups, or modules
6. Explains the overall structure and organization
7. Highlights any specialized components or connections
8. Notes any directionality or weighted relationships
9. Connects this network to its functional purpose

Your description should be educational, technical, and thorough for students preparing for exams.
""",

        'architecture_diagram': """
This architecture diagram appears in educational material. Provide a detailed explanation that:

1. Identifies the type of system architecture being shown
2. Describes each component or layer and its function
3. Explains how components interact and data flows between them
4. Identifies key interfaces or connection points
5. Describes the hierarchical structure or organization
6. Explains any redundancy, security, or scaling features
7. Identifies external systems or user interaction points
8. Notes any constraints, dependencies, or requirements
9. Connects this architecture to its intended purpose

Your description should be educational, technical, and thorough for students preparing for exams.
""",

        'uml_diagram': """
This UML diagram appears in educational material. Provide a detailed explanation that:

1. Identifies the specific type of UML diagram (class, sequence, activity, etc.)
2. Describes each element in the diagram (classes, methods, relationships, etc.)
3. Explains inheritance hierarchies or object relationships
4. Describes the attributes and operations of key classes
5. Explains the significance of different relationship types
6. Identifies design patterns if applicable
7. Describes sequence or interaction flow for behavioral diagrams
8. Notes any constraints, conditions, or special notations
9. Connects this design to software engineering principles

Your description should be educational, technical, and thorough for students preparing for exams.
""",

        'er_diagram': """
This Entity-Relationship diagram appears in educational material. Provide a detailed explanation that:

1. Identifies the database or data domain being modeled
2. Describes each entity and its purpose
3. Explains the attributes of each entity, noting primary keys
4. Describes relationships between entities with cardinality (one-to-many, etc.)
5. Identifies any weak entities or associative entities
6. Explains the overall data structure and organization
7. Notes any constraints, dependencies, or business rules
8. Identifies database normalization level if applicable
9. Connects this data model to its functional purpose

Your description should be educational, technical, and thorough for students preparing for exams.
""",

        # Additional specialized diagram types
        'organizational_chart': """
This organizational chart appears in educational material. Provide a detailed explanation that:

1. Identifies the organization or department being depicted
2. Describes the hierarchical structure from top to bottom
3. Explains the roles and responsibilities at each level
4. Identifies reporting relationships and chains of command
5. Describes any functional divisions or departments
6. Notes spans of control or management structure
7. Explains any matrix or non-traditional relationships
8. Describes the overall organizational design principles
9. Connects this structure to organizational theory concepts

Your description should be educational, focused on organizational design, and provide insights a student would need for exam preparation.
""",

        'concept_map': """
This concept map appears in educational material. Provide a detailed explanation that:

1. Identifies the central concept or topic being mapped
2. Describes the key concepts and their relationships
3. Explains how concepts connect or build upon each other
4. Identifies hierarchical or causal relationships
5. Describes groupings of related concepts
6. Notes cross-links between different branches
7. Explains the overall knowledge structure being represented
8. Identifies foundational vs. advanced concepts
9. Connects this conceptual framework to the broader subject area

Your description should be educational, conceptually focused, and provide insights a student would need for exam preparation.
""",

        'decision_tree': """
This decision tree appears in educational material. Provide a detailed explanation that:

1. Identifies the decision process or classification system being depicted
2. Describes the root node and initial decision point
3. Explains each branch and decision criterion
4. Traces paths from root to leaf nodes
5. Describes the outcomes or classifications at leaf nodes
6. Explains the decision logic and sequence
7. Identifies key decision factors or variables
8. Notes any probabilistic elements or weights
9. Connects this decision model to its practical application

Your description should be educational, decision-focused, and provide insights a student would need for exam preparation.
"""
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
        self.vision_adapter = None
        self.vision_provider = "auto"
        self.cache = {}

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

            # Handle vision settings
            self.vision_provider = config.get('vision_provider', 'auto')
            self.vision_model = config.get('vision_model', None)
            self.vision_api_key = config.get('vision_api_key', None)
            self.vision_base_url = config.get('vision_base_url', None)

            # Caching settings
            self.enable_caching = config.get('enable_diagram_description_cache', True)
            self.cache_expiry = config.get('diagram_description_cache_expiry', 3600 * 24 * 7)  # Default: 1 week
            self.cache_dir = config.get('diagram_description_cache_dir', os.path.join(os.path.expanduser('~'), '.lightrag', 'diagram_cache'))

            # Initialize cache if enabled
            if self.enable_caching:
                self._init_cache()

    def _import_vision_registry(self):
        """Helper method to import vision registry (makes testing easier)."""
        from lightrag.llm.vision_adapter import vision_registry
        return vision_registry

    async def initialize_vision_adapter(self) -> bool:
        """
        Initialize the vision adapter based on configuration with fallback.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Import vision adapter module
            vision_registry = self._import_vision_registry()

            # Track attempted providers to avoid repeated failures
            attempted_providers = []

            if self.vision_provider.lower() == "auto":
                # Try to find the best available adapter
                self.vision_adapter = await vision_registry.find_best_available_adapter()
                if self.vision_adapter:
                    logger.info(f"Automatically selected vision provider: {self.vision_adapter.provider_name}")
                    return True
                else:
                    logger.warning("No vision adapters available through auto-selection. Will try explicit providers.")
            else:
                # Use the specified provider
                self.vision_adapter = await vision_registry.get_adapter(
                    self.vision_provider,
                    api_key=self.vision_api_key,
                    model_name=self.vision_model,
                    base_url=self.vision_base_url
                )

                if self.vision_adapter:
                    logger.info(f"Initialized vision adapter for provider: {self.vision_adapter.provider_name}")
                    return True
                else:
                    logger.warning(f"Failed to initialize vision adapter for {self.vision_provider}. Will attempt fallbacks.")
                    attempted_providers.append(self.vision_provider.lower())

            # If we reach here, we need to try fallback providers
            # Try each provider in order of reliability
            fallback_providers = ["anthropic", "openai"]

            for provider in fallback_providers:
                # Skip if we already tried this provider
                if provider.lower() in attempted_providers:
                    continue

                logger.info(f"Attempting fallback vision provider: {provider}")
                attempted_providers.append(provider.lower())

                try:
                    self.vision_adapter = await vision_registry.get_adapter(provider)

                    if self.vision_adapter:
                        logger.info(f"Successfully initialized fallback vision adapter: {self.vision_adapter.provider_name}")
                        return True
                    else:
                        logger.warning(f"Failed to initialize fallback vision adapter for {provider}")
                except Exception as provider_e:
                    logger.warning(f"Error initializing fallback vision adapter for {provider}: {str(provider_e)}")

            # If we've tried all providers and none worked
            logger.warning("All vision providers failed. Diagram descriptions will be limited.")
            return False

        except ImportError as e:
            logger.warning(f"Vision adapter module not available: {str(e)}. Diagram descriptions will be limited.")
            return False
        except Exception as e:
            logger.error(f"Error initializing vision adapter: {str(e)}")
            return False

    def _init_cache(self):
        """Initialize the cache for diagram descriptions."""
        try:
            # Create cache directory if it doesn't exist
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

            # Get the cache file path
            self.cache_file = os.path.join(self.cache_dir, 'diagram_descriptions.pkl')

            # Load existing cache if available
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'rb') as f:
                        loaded_cache = pickle.load(f)
                        if isinstance(loaded_cache, dict):
                            self.cache = loaded_cache

                            # Clean expired entries
                            self._clean_cache()

                            num_entries = len(self.cache)
                            logger.info(f"Loaded {num_entries} diagram descriptions from cache")
                        else:
                            logger.warning(f"Cache file exists but contains invalid data. Using empty cache.")
                            self.cache = {}
                except Exception as e:
                    logger.warning(f"Error loading cache file: {str(e)}. Using empty cache.")
                    self.cache = {}
            else:
                logger.debug("No cache file found. Using empty cache.")
                self.cache = {}
        except Exception as e:
            logger.warning(f"Error initializing cache: {str(e)}. Caching will be disabled.")
            self.enable_caching = False
            self.cache = {}

    def _clean_cache(self):
        """Remove expired entries from the cache."""
        if not self.enable_caching:
            return

        now = time.time()
        expired_keys = []

        for key, entry in self.cache.items():
            if 'timestamp' in entry:
                age = now - entry['timestamp']
                if age > self.cache_expiry:
                    expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired entries from diagram description cache")

    def _save_cache(self):
        """Save the current cache to disk."""
        if not self.enable_caching:
            return

        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} entries to diagram description cache")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")

    def clear_cache(self):
        """Clear the diagram description cache."""
        self.cache = {}
        if self.enable_caching:
            try:
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                logger.info("Diagram description cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing cache file: {str(e)}")

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

    async def generate_diagram_description(self, diagram_data: Dict[str, Any], diagram_type: str = 'general') -> str:
        """
        Generates a textual description for a diagram using vision models.

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
        diagram_id = diagram_data.get('diagram_id', 'unknown')
        page_num = diagram_data.get('page', 'unknown')

        if not image_data:
            logger.error("Missing '_full_base64' in diagram data.")
            return "Error: Missing image data."

        # Check if we have a cached description for this diagram
        if self.enable_caching and diagram_id in self.cache:
            cached_result = self.cache[diagram_id]
            if cached_result.get('diagram_type') == diagram_type:
                logger.info(f"Using cached description for diagram {diagram_id}")
                return cached_result.get('description')

        # Create the prompt for description
        prompt = self._create_description_prompt(diagram_type=diagram_type)
        diagram_data['diagram_type'] = diagram_type
        logger.debug(f"Generating description for diagram {diagram_id} type: {diagram_type}")

        # Add context to the prompt if available
        context_text = ""
        if caption:
            context_text += f"Caption: \"{caption}\"\n\n"
        if surrounding_text and (not caption or caption not in surrounding_text):
            # Clean up surrounding text - trim to reasonable length and remove excessive whitespace
            max_context_length = 1000
            if len(surrounding_text) > max_context_length:
                # Take the middle portion centered around where the diagram appears
                start_pos = max(0, len(surrounding_text) // 2 - max_context_length // 2)
                surrounding_text = surrounding_text[start_pos:start_pos + max_context_length]
                surrounding_text = f"... {surrounding_text} ..."

            # Clean up and normalize whitespace
            surrounding_text = re.sub(r'\s+', ' ', surrounding_text).strip()
            context_text += f"Context from document: {surrounding_text}\n\n"

        # Add page information
        context_text += f"Page: {page_num}\n\n"

        # Combine context with prompt
        full_prompt = f"{context_text}{prompt}"

        # Decode the base64 image
        try:
            img_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Error decoding base64 image for diagram {diagram_id}: {str(e)}")
            return f"Error decoding image: {str(e)}"

        # Try using our vision adapter
        if not self.vision_adapter:
            # Try to initialize the vision adapter
            try:
                init_success = await self.initialize_vision_adapter()
                if not init_success:
                    logger.warning("Failed to initialize vision adapter. Using fallback description method.")
            except Exception as e:
                logger.error(f"Error initializing vision adapter: {str(e)}")

        if self.vision_adapter:
            try:
                # Build a context dict with additional information
                vision_context = {
                    "system_prompt": f"You are an educational AI that provides detailed explanations of diagrams for students. You are analyzing a diagram extracted from a document on page {page_num}.",
                    "max_tokens": 1500,
                    "temperature": 0.3,
                    "image_detail": "high"
                }

                # Call the vision adapter with the image and prompt
                description = await self.vision_adapter.generate_description(
                    image_data=img_bytes,
                    prompt=full_prompt,
                    context=vision_context
                )

                # Cache the result if caching is enabled
                if self.enable_caching:
                    self.cache[diagram_id] = {
                        'diagram_type': diagram_type,
                        'description': description,
                        'timestamp': time.time(),
                        'provider': self.vision_adapter.provider_name
                    }
                    # Save the cache to disk
                    self._save_cache()

                return description

            except Exception as e:
                logger.error(f"Error generating diagram description with vision adapter: {str(e)}")
                # Fall through to alternative methods

        # If we have an LLM service, use it as a fallback
        if self.llm_service:
            try:
                # Try to create a text-based prompt for the LLM service
                text_prompt = f"""
                You are analyzing a diagram that was found on page {page_num} of a document.

                Your task is to provide your best analysis based on the context information below:
                """

                # Add caption if available
                if caption:
                    text_prompt += f"\nCaption of the diagram: \"{caption}\"\n"

                # Add surrounding text if available
                if surrounding_text:
                    text_prompt += f"\nText surrounding the diagram: \"{surrounding_text}\"\n"

                # Add diagram dimensions if available
                if diagram_data.get('width') and diagram_data.get('height'):
                    text_prompt += f"\nDimensions: {diagram_data.get('width')}x{diagram_data.get('height')} pixels\n"

                # Add instructions
                text_prompt += """
                Based only on this context information, please provide your best educated guess about:
                1. What type of diagram this might be
                2. What key concepts or processes it might be illustrating
                3. How it relates to the surrounding text or caption

                Note: You cannot see the actual diagram, so your response should clearly indicate that this is an inference based on limited information.
                """

                # Call the LLM service
                if hasattr(self.llm_service, 'generate_text') and callable(self.llm_service.generate_text):
                    text_response = await self.llm_service.generate_text(text_prompt)
                elif callable(self.llm_service):
                    text_response = await self.llm_service(text_prompt)
                else:
                    # Default response if LLM service interface is unknown
                    text_response = f"This diagram appears on page {page_num}. "
                    if caption:
                        text_response += f"The caption indicates it is about: {caption}. "
                    text_response += "Without access to the image itself, a detailed description cannot be provided."

                # Cache the fallback response if enabled
                if self.enable_caching:
                    self.cache[diagram_id] = {
                        'diagram_type': diagram_type,
                        'description': text_response,
                        'timestamp': time.time(),
                        'provider': 'text_llm_fallback'
                    }
                    # Save the cache to disk
                    self._save_cache()

                return text_response

            except Exception as e:
                logger.error(f"Error generating diagram description with LLM service: {str(e)}")
                # Fall through to basic description

        # If all else fails, return a basic description
        basic_desc = f"Diagram on page {page_num}"
        if caption:
            basic_desc += f" with caption: {caption}"
        if diagram_data.get('width') and diagram_data.get('height'):
            basic_desc += f". Dimensions: {diagram_data.get('width')}x{diagram_data.get('height')} pixels"

        return basic_desc

    def generate_diagram_description_sync(self, diagram_data: Dict[str, Any], diagram_type: str = 'general') -> str:
        """
        Synchronous wrapper for generate_diagram_description.

        This method exists for backward compatibility with code that expects a synchronous interface.

        Args:
            diagram_data (Dict[str, Any]): Diagram data including base64 image
            diagram_type (str, optional): Type of diagram. Defaults to 'general'.

        Returns:
            str: Textual description of the diagram
        """
        import asyncio

        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async method
        return loop.run_until_complete(self.generate_diagram_description(diagram_data, diagram_type))

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