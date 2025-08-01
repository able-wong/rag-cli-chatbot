#!/bin/bash

# doit.sh - A simple script for common development tasks



case "$1" in
    test)
        echo "Running tests..."
        pytest tests/
        ;;
    integration-test)
        echo "Running integration tests..."
        echo "Note: Integration tests require a working config.yaml with valid LLM provider settings"
        if [ ! -f "config/config.yaml" ]; then
            echo "Error: config/config.yaml not found. Copy config/config.sample.yaml and configure your LLM settings."
            exit 1
        fi
        cd integration_tests && pytest -c pytest.ini
        ;;
    lint)
        echo "Running linter (ruff) without applying fixes..."
        ruff check . --exclude venv
        ;;
    lint-fix)
        echo "Running linter (ruff) and applying fixes..."
        ruff check . --fix --exclude venv
        ;;
    *)
        echo "Usage: ./doit.sh {test|integration-test|lint|lint-fix}"
        exit 1
        ;;
esac