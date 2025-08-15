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
    check-query-rewriter)
        if [ -z "$2" ]; then
            echo "Usage: ./doit.sh check-query-rewriter 'your query here'"
            echo "Example: ./doit.sh check-query-rewriter 'search @knowledgebase on AI, only by Smith'"
            exit 1
        fi
        echo "Testing QueryRewriter with query: $2"
        python check_query_rewriter.py "$2"
        ;;
    *)
        echo "Usage: ./doit.sh {test|integration-test|lint|lint-fix|check-query-rewriter}"
        echo ""
        echo "Commands:"
        echo "  test                      Run unit tests"
        echo "  integration-test          Run integration tests (need working config.yaml)"
        echo "  lint                      Run linter check"
        echo "  lint-fix                  Run linter and apply fixes"
        echo "  check-query-rewriter      Test query rewriter with a specific query"
        echo ""
        echo "Examples:"
        echo "  ./doit.sh test"
        echo "  ./doit.sh check-query-rewriter 'search @knowledgebase on AI, only by Smith'"
        exit 1
        ;;
esac
