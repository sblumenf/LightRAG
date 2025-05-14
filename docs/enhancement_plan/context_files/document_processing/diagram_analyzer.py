"""
Diagram analysis module for GraphRAG tutor.
"""
import logging
import os
import base64
import tempfile
import uuid
import io
from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np

# Try to import necessary libraries with graceful fallbacks
logger = logging.getLogger(__name__) # Define logger early for warnings

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Installing it is recommended for diagram extraction.")

try:
    import PIL
    from PIL import Image, ImageFilter # Added import for ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Image processing capabilities will be limited.")

try:
    import cv2 # Added import for OpenCV
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Shape detection capabilities will be limited.")


try:
    from google import generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI library not available. Vision capabilities will be limited.")

# Import settings *after* defining logger and availability flags
try:
    from config import settings
except ImportError:
    # Define fallback settings if config/settings.py is missing or causes issues
    class FallbackSettings:
        DEFAULT_DIAGRAM_PROMPTS = {'general': "Describe this diagram."}
        DIAGRAM_DETECTION_THRESHOLD = 0.6
        LLM_PROVIDER = "gemini" # Default assumption
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Still try to get from env
    settings = FallbackSettings()
    logger.warning("Could not import config.settings, using fallback defaults.")


class DiagramAnalyzer:
    """
    Extract and analyze diagrams from documents, generating textual descriptions.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_service=None, api_key: Optional[str] = None, model_name: str = 'gemini-pro-vision'): # Add config parameter with type hint
        """
        Initialize the diagram analyzer.

        Args:
            config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
            llm_service: Optional LLM service for diagram description
            api_key: Optional API key for vision model (Gemini)
            model_name: Vision model name to use
        """
        self.llm_service = llm_service
        # Determine API key (use arg, then config, then settings/env)
        _config_api_key = config.get('api_key') if config else None
        self.api_key = api_key or _config_api_key or settings.GOOGLE_API_KEY

        self.model_name = model_name
        self.vision_model = None
        # Load initial prompts from settings and copy to avoid modification
        self.description_prompts = settings.DEFAULT_DIAGRAM_PROMPTS.copy()
        # Default threshold from settings
        self.diagram_detection_threshold = settings.DIAGRAM_DETECTION_THRESHOLD

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


        # Initialize vision model if Gemini is available and API key is provided
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.vision_model = genai.GenerativeModel(model_name)
                logger.info(f"Initialized vision model: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing vision model: {str(e)}")
        elif settings.LLM_PROVIDER.lower() == "gemini" and not self.api_key:
             logger.warning("Gemini selected as LLM provider, but no API key provided.")


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
        else: shape_score = 0.0

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
        # --- DEBUG LOG ---
        logger.debug(f"Checking diagram status: Score={diagram_score:.2f}, Threshold={self.diagram_detection_threshold}, IsDiagram={is_diagram}")
        # --- END DEBUG LOG ---
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

    def _generate_with_gemini(self, image_data: str, prompt: str) -> str:
        """Generate a description using Gemini vision model."""
        if not GEMINI_AVAILABLE or not self.vision_model:
            logger.error("Gemini vision model not available or initialized.")
            return "Error: Vision model not available"
        try:
            image_bytes = base64.b64decode(image_data)
            content = [prompt, {"mime_type": "image/png", "data": image_bytes}]
            logger.debug(f"Sending prompt to Gemini: {prompt[:100]}...")
            response = self.vision_model.generate_content(content)
            if not response.candidates:
                 logger.warning("Gemini response has no candidates.")
                 try: block_reason = response.prompt_feedback.block_reason; logger.warning(f"Gemini blocked. Reason: {block_reason}"); return f"Error: Blocked ({block_reason})."
                 except Exception: return "Error: No candidates."
            try: # Safety check
                safety_ratings = response.candidates[0].safety_ratings
                if any(rating.probability in ['MEDIUM', 'HIGH'] for rating in safety_ratings): logger.warning(f"Unsafe content detected: {safety_ratings}")
            except Exception as e: logger.warning(f"Could not check safety ratings: {e}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating with Gemini: {str(e)}", exc_info=True)
            if "API key not valid" in str(e): return "Error: Invalid Gemini API Key."
            return f"Error generating description: {str(e)}"

    def _generate_with_custom_llm(self, image_path: str, prompt: str) -> str:
        """Generate a description using a custom LLM service."""
        if not self.llm_service: logger.error("Custom LLM service not configured."); return "Error: LLM service not available"
        try:
            if not os.path.exists(image_path): logger.error(f"Image path not found for custom LLM: {image_path}"); return "Error: Image file not found."
            with open(image_path, 'rb') as img_file: image_data_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            logger.debug(f"Sending prompt to custom LLM: {prompt[:100]}...")
            response = self.llm_service.generate_with_image(prompt=prompt, image_base64=image_data_base64)
            return response.text.strip()
        except AttributeError: logger.error("Custom LLM interface mismatch."); return "Error: Custom LLM interface mismatch."
        except Exception as e: logger.error(f"Error generating with custom LLM: {str(e)}", exc_info=True); return f"Error generating description: {str(e)}"

    def extract_diagrams_from_pdf(self, pdf_path: str, diagram_type: str = 'general') -> List[Dict[str, Any]]:
        """Extract diagrams from a PDF document."""
        if not PYMUPDF_AVAILABLE: logger.warning("PyMuPDF not available."); return []
        if not os.path.exists(pdf_path): logger.error(f"PDF file not found: {pdf_path}"); return []

        extracted_diagrams = []; min_size = 100; pdf = None

        try:
            with fitz.open(pdf_path) as pdf:
                logger.info(f"Processing PDF: {pdf_path} ({len(pdf)} pages)")

                for page_index, page in enumerate(pdf):
                    page_num = page_index + 1; page_width, page_height = page.rect.width, page.rect.height
                    logger.debug(f"Processing Page {page_num}/{len(pdf)}")
                    image_list = page.get_images(full=True)
                    logger.debug(f"Page {page_num}: Found {len(image_list)} raw image objects.")
                    processed_xrefs = set()

                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        if xref in processed_xrefs: continue
                        try:
                            base_image = pdf.extract_image(xref)
                            if not base_image or not base_image.get("image"): logger.warning(f"P{page_num},I{img_index}({xref}):Failed extract"); continue
                            image_data = base_image["image"]; image_ext = base_image["ext"]; image_format = base_image.get("colorspace", "unknown")
                            width, height = 0, 0
                            if PIL_AVAILABLE:
                                try: img_pil = Image.open(io.BytesIO(image_data)); width, height = img_pil.size
                                except Exception as pil_e: logger.warning(f"P{page_num},I{img_index}({xref}):PIL fail:{pil_e}.PDF dims."); width, height = base_image.get("width", 0), base_image.get("height", 0)
                            else: width, height = base_image.get("width", 0), base_image.get("height", 0)
                            logger.debug(f"P{page_num},I{img_index}({xref}):Extracted {image_ext}, Size {width}x{height}")

                            # --- DEBUG LOG BEFORE SIZE CHECK ---
                            logger.debug(f"Checking image size: {width}x{height} against min_size {min_size}")
                            if width < min_size or height < min_size: logger.debug(f"Skipping small image ({width}x{height})."); continue
                            # --- END DEBUG LOG ---

                            # --- DEBUG LOG BEFORE _is_diagram CALL ---
                            logger.debug(f"Calling _is_diagram for xref {xref}")
                            is_diagram = self._is_diagram(image_data, {'page_width': page_width, 'page_height': page_height, 'format': image_format})
                            # --- END DEBUG LOG ---

                            if is_diagram:
                                logger.info(f"P{page_num},I{img_index}({xref}): Identified as diagram.")
                                img_id = f"img-{uuid.uuid4()}"; image_b64 = base64.b64encode(image_data).decode('utf-8')
                                image_rect_list = None # Initialize as None
                                try:
                                    # get_image_bbox returns a Rect object
                                    bbox = page.get_image_bbox(img_info, transform=False)
                                    if bbox:
                                        # Convert fitz.Rect to list [x0, y0, x1, y1]
                                        image_rect_list = [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                                        logger.debug(f"P{page_num},I{img_index}({xref}): Found bbox: {image_rect_list}")
                                    else: logger.warning(f"P{page_num},I{img_index}({xref}): No bbox from get_image_bbox.")
                                except Exception as rect_e: logger.warning(f"P{page_num},I{img_index}({xref}): Error get bbox: {rect_e}")
                                surrounding_text = ""
                                if image_rect_list:
                                    try:
                                        expansion = 50; text_rect = fitz.Rect(max(0, image_rect_list[0] - expansion), max(0, image_rect_list[1] - expansion), min(page_width, image_rect_list[2] + expansion), min(page_height, image_rect_list[3] + expansion))
                                        surrounding_text = page.get_text("text", clip=text_rect).strip()
                                        logger.debug(f"P{page_num},I{img_index}({xref}): Text len {len(surrounding_text)}.")
                                    except Exception as text_e: logger.warning(f"P{page_num},I{img_index}({xref}): Error text: {text_e}")
                                diagram_entry = {'diagram_id': img_id, 'page': page_num, 'width': width, 'height': height, 'format': image_ext, 'is_diagram': True, 'position': image_rect_list, 'surrounding_text': surrounding_text, 'caption': None, 'file_path': None, '_full_base64': image_b64, 'base64_data': image_b64[:100] + '...', 'description': None, 'extraction_method': 'pymupdf_image', 'diagram_type': diagram_type}
                                extracted_diagrams.append(diagram_entry); processed_xrefs.add(xref)
                            else: logger.debug(f"P{page_num},I{img_index}({xref}): Skipped non-diagram.")
                        except Exception as img_proc_e: logger.warning(f"P{page_num}: Error processing img xref {xref}: {img_proc_e}")

                    try: # Caption processing
                        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
                        for block in blocks:
                            if block["type"] == 0:
                                block_text = "".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
                                caption_match = re.match(r'(Figure|Fig\.?|Table)\s+(\d+[\.\d]*)\s*[:.-]?\s*(.*)', block_text, re.IGNORECASE)
                                if caption_match:
                                    caption_text = block_text; caption_rect = fitz.Rect(block["bbox"])
                                    logger.debug(f"P{page_num}: Potential caption: '{caption_text[:50]}...'")
                                    closest_diagram = None; min_distance = float('inf')
                                    for diagram in extracted_diagrams:
                                        if diagram['page'] == page_num and diagram['position']:
                                            img_rect = fitz.Rect(diagram['position'])
                                            vertical_dist = min(abs(caption_rect.y0 - img_rect.y1), abs(caption_rect.y1 - img_rect.y0))
                                            h_overlap = max(0, min(caption_rect.x1, img_rect.x1) - max(caption_rect.x0, img_rect.x0))
                                            if vertical_dist < 100 and h_overlap > 0:
                                                distance = vertical_dist + (img_rect.width - h_overlap) * 0.1
                                                if distance < min_distance: min_distance = distance; closest_diagram = diagram
                                    if closest_diagram: logger.info(f"P{page_num}: Assoc caption '{caption_text[:50]}...' w/ diagram {closest_diagram['diagram_id']}"); closest_diagram['caption'] = caption_text
                                    else: logger.debug(f"P{page_num}: Caption '{caption_text[:50]}...' no match.")
                    except Exception as caption_e: logger.warning(f"P{page_num}: Error captions: {caption_e}", exc_info=True)

                logger.info(f"Finished PDF. Extracted {len(extracted_diagrams)} diagrams.")
                return extracted_diagrams
        except Exception as e:
            logger.error(f"Fatal error extracting diagrams from PDF '{pdf_path}': {str(e)}", exc_info=True)
            return []

    def generate_diagram_description(self, diagram_data: Dict[str, Any], diagram_type: str = 'general') -> str:
        """Generates a textual description for a diagram."""
        image_data = diagram_data.get('_full_base64'); surrounding_text = diagram_data.get('surrounding_text'); caption = diagram_data.get('caption')
        if not image_data: logger.error("Missing '_full_base64'."); return "Error: Missing image data."
        prompt = self._create_description_prompt(diagram_type=diagram_type)
        diagram_data['diagram_type'] = diagram_type; logger.debug(f"Generating desc for diagram {diagram_data.get('diagram_id', 'N/A')} type: {diagram_type}")
        context = "";
        if caption: context += f"Caption: \"{caption}\"\n\n"
        if surrounding_text and (not caption or caption not in surrounding_text): context += f"Context: {surrounding_text}\n\n"
        full_prompt = f"{context}{prompt}"
        llm_provider = settings.LLM_PROVIDER.lower(); logger.info(f"Using LLM provider: {llm_provider}")
        if llm_provider == "gemini": return self._generate_with_gemini(image_data=image_data, prompt=full_prompt)
        elif llm_provider == "custom":
            image_path = diagram_data.get('file_path'); temp_file_created = False
            if not image_path:
                 try:
                     img_bytes=base64.b64decode(image_data); img_format=diagram_data.get('format','png').lower(); suffix=f".{img_format}" if img_format in ['png','jpg','jpeg','gif','bmp'] else ".png"
                     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_img: temp_img.write(img_bytes); image_path = temp_img.name; temp_file_created = True
                     logger.info(f"Saved temporary image: {image_path}")
                 except Exception as e: logger.error(f"Error creating temp image: {e}"); return "Error: Could not process image for custom LLM."
            description = self._generate_with_custom_llm(image_path=image_path, prompt=full_prompt)
            if temp_file_created and image_path and os.path.exists(image_path):
                 try: os.remove(image_path); logger.info(f"Removed temporary image: {image_path}")
                 except Exception as e: logger.warning(f"Could not remove temp image {image_path}: {e}")
            return description
        else: logger.error(f"LLM Provider '{settings.LLM_PROVIDER}' unsupported."); return "Error: LLM Provider not configured."