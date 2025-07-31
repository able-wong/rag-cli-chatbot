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

## Post-Task Actions

After completing a requested task (e.g., code modification, file operation), you should:

1.  **Offer to run tests**: Ask the user if they would like you to run the project's tests using `./doit.sh test`.
2.  **Run linter and auto-fix**: Automatically run the linter with auto-fix using `./doit.sh lint-fix`. If issues remain, report them to the user.
3.  **Offer to commit changes**: If changes were made (either by the task or by `lint-fix`), ask the user if they would like you to commit the changes. If so, propose a draft commit message.
