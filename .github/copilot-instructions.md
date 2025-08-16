# AI Assistant/Agent Instructions

This document provides instructions for interacting with the `rag-cli-chatbot` project.

## Project Setup and Configuration

For detailed project setup, installation, and configuration instructions, refer to the **README.md** file which contains:
- Prerequisites and installation steps
- LLM provider configuration (Ollama, Gemini)
- Embedding provider setup (SentenceTransformers, Ollama, Gemini)
- Sparse embedding configuration for hybrid search (SPLADE)
- Vector database setup (Qdrant local/cloud)
- RAG retrieval strategies and advanced features

**Quick Reference**: Main entry point is `python3 main.py` (requires virtual environment activation).

## Running Python Scripts

To run any Python script within this project (e.g., `main.py`), you must also activate the virtual environment first.

Example:

```bash
source venv/bin/activate && python3 src/main.py
```

or running tests

Always ensure the virtual environment is activated to use the correct dependencies.

```bash
source venv/bin/activate && ./doit.sh test
```

## Available doit.sh Commands

The project includes a `doit.sh` script for common development tasks:

- `./doit.sh test` - Run unit tests
- `./doit.sh integration-test` - Run integration tests (need working config.yaml)
- `./doit.sh lint` - Run linter check
- `./doit.sh lint-fix` - Run linter and apply fixes
- `./doit.sh check-query-rewriter 'query'` - Test query rewriter with a specific query

Examples:
```bash
./doit.sh test
./doit.sh lint-fix
./doit.sh check-query-rewriter 'search @knowledgebase on AI, only by Smith'
```

## Post-Task Actions

After completing a requested task (e.g., code modification, file operation), you should:

1.  **Offer to run tests**: Ask the user if they would like you to run the project's tests using `./doit.sh test`.
2.  **Run linter and auto-fix**: Automatically run the linter with auto-fix using `./doit.sh lint-fix`. If issues remain, report them to the user.
3.  **Offer to commit changes**: If changes were made (either by the task or by `lint-fix`), ask the user if they would like you to commit the changes. If so, propose a draft commit message.

## Integration Tests

- **Full suite**: `./doit.sh integration-test` - Runs all integration tests but is slow (>4 mins) and incurs API costs. Ask user first.
- **Specific tests**: Run only necessary integration test cases when possible for faster feedback:
  - `pytest integration_tests/test_query_rewriter_integration.py -v` - Intent-based filtering & query cleaning
  - `pytest integration_tests/test_query_rewriter_multi_persona_hyde.py -v` - HyDE multi-persona functionality  
  - `pytest integration_tests/test_search_service_integration.py -v` - SearchService hybrid search workflows
  - `pytest integration_tests/test_qdrant_integration.py -v` - Vector database operations
  - `pytest integration_tests/test_cli_integration.py -v` - CLI workflow testing
- **Important**: Integration tests must use `ConfigManager` for configuration loading to properly handle environment variables (like `GEMINI_API_KEY`). If the chatbot app works, integration tests should work too.
- Integration tests that fail to connect should skip gracefully with appropriate messages.

## Note

- When creating new integration tests, always use `ConfigManager` instead of directly loading YAML files to ensure proper environment variable handling.
