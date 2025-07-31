# GitHub Copilot Instructions

This document provides instructions for interacting with the `rag-cli-chatbot` project.

## Running Tests

To run the test suite, you must first activate the Python virtual environment and then run `pytest`. Use the following command from the project root:

```bash
source venv/bin/activate && pytest
```

## Running Python Scripts

To run any Python script within this project (e.g., `main.py`), you must also activate the virtual environment first.

Example:
```bash
source venv/bin/activate && python3 src/cli.py
```

Always ensure the virtual environment is activated to use the correct dependencies.
