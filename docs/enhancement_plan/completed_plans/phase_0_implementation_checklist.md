# Phase 0 Implementation Checklist

## Task 0.1: Configuration Management for Schema & New Features

- [x] Add configuration for path to existing `schema.json` file
  - [x] Add `SCHEMA_FILE_PATH` to `.env` file
  - [x] Add `SCHEMA_FILE_PATH` to `.env.example` file
  - [x] Create configuration loader for schema path

- [x] Add boolean flags for enabling/disabling diagram/formula analysis
  - [x] Add `ENABLE_DIAGRAM_ANALYSIS` to `.env` file
  - [x] Add `ENABLE_DIAGRAM_ANALYSIS` to `.env.example` file
  - [x] Add `ENABLE_FORMULA_ANALYSIS` to `.env` file
  - [x] Add `ENABLE_FORMULA_ANALYSIS` to `.env.example` file
  - [x] Create configuration loader for diagram/formula analysis flags

- [x] Add boolean flag for Chain-of-Thought (CoT) enablement
  - [x] Add `ENABLE_COT` to `.env` file
  - [x] Add `ENABLE_COT` to `.env.example` file
  - [x] Create configuration loader for CoT flag

- [x] Add configurable thresholds for entity resolution similarity scores
  - [x] Add `ENTITY_RESOLUTION_NAME_THRESHOLD` to `.env` file
  - [x] Add `ENTITY_RESOLUTION_NAME_THRESHOLD` to `.env.example` file
  - [x] Add `ENTITY_RESOLUTION_EMBEDDING_THRESHOLD` to `.env` file
  - [x] Add `ENTITY_RESOLUTION_EMBEDDING_THRESHOLD` to `.env.example` file
  - [x] Add `ENTITY_RESOLUTION_CONTEXT_THRESHOLD` to `.env` file
  - [x] Add `ENTITY_RESOLUTION_CONTEXT_THRESHOLD` to `.env.example` file
  - [x] Add `ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY` to `.env` file
  - [x] Add `ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY` to `.env.example` file
  - [x] Create configuration loader for entity resolution thresholds

- [x] Create configuration loader module
  - [x] Implement `config_loader.py` with `EnhancedConfig` class
  - [x] Implement singleton pattern for configuration
  - [x] Add proper error handling for invalid values

- [x] Update LightRAG class to use new configuration
  - [x] Add schema file path field
  - [x] Add diagram/formula analysis flag fields
  - [x] Add CoT flag field
  - [x] Add entity resolution threshold fields

- [x] Write tests for configuration loading
  - [x] Test default values
  - [x] Test loading from environment variables
  - [x] Test handling of invalid values
  - [x] Test integration with LightRAG class

## Task 0.2: Schema Loading Utility

- [x] Implement schema loading utility
  - [x] Create `load_schema(schema_path: str) -> dict` function
  - [x] Add path normalization
  - [x] Add file existence check
  - [x] Add JSON parsing
  - [x] Add robust error handling

- [x] Write tests for schema loading utility
  - [x] Test loading schema from valid path
  - [x] Test loading schema from nonexistent path
  - [x] Test loading schema with invalid JSON
  - [x] Test loading schema with empty path
  - [x] Test handling of general exceptions

- [x] Test integration with configuration
  - [x] Test loading schema using path from configuration

## Task 0.3: Basic Test Setup

- [x] Create pytest configuration
  - [x] Create `pytest.ini` file
  - [x] Configure test paths
  - [x] Configure test file patterns
  - [x] Configure asyncio marker

- [x] Create test fixtures
  - [x] Create fixture for sample text file loading
  - [x] Create fixture for schema loading
  - [x] Create fixture for temporary working directory
  - [x] Create fixture for LightRAG instance initialization
  - [x] Create fixture for LightRAG instance with sample document

- [x] Create sample test data
  - [x] Create `tests/fixtures/sample_doc.txt`

- [x] Write tests for fixtures
  - [x] Test sample document path fixture
  - [x] Test sample document content fixture
  - [x] Test schema path fixture
  - [x] Test schema fixture
  - [x] Test temporary working directory fixture
  - [x] Test LightRAG instance fixture
  - [x] Test LightRAG instance with sample document fixture

## Verification

- [x] All tests pass without warnings or skips
- [x] 100% code coverage for new modules
- [x] Configuration is properly loaded into LightRAG class
- [x] Schema can be loaded using path from configuration
- [x] LightRAG server can start with new configuration
