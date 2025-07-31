#!/bin/bash

# doit.sh - A simple script for common development tasks



case "$1" in
    test)
        echo "Running tests..."
        pytest
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
        echo "Usage: ./doit.sh {test|lint|lint-fix}"
        exit 1
        ;;
esac