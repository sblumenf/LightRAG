"""
Configuration settings for the LLM generator.
"""

class Settings:
    """Settings class for the LLM generator."""
    
    def __init__(self):
        """Initialize settings with default values."""
        self.openai_api_key = "test_api_key"
        self.gemini_api_key = "test_api_key"
        self.default_llm_provider = "openai"
        self.default_llm_model = "gpt-4"
        self.max_refinement_attempts = 2
        self.enable_cot = True
        self.debug_mode = False
        self.log_level = "INFO"

# Create a singleton instance
settings = Settings()
