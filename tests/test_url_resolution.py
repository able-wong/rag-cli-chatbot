#!/usr/bin/env python3
"""
Unit tests for URL resolution functionality in the RAG CLI.
Tests the _resolve_source_url method with various URL formats.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli import RAGCLI


class TestURLResolution(unittest.TestCase):
    """Test cases for URL resolution functionality."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        # Mock all the heavy dependencies
        with patch('cli.ConfigManager'), \
             patch('cli.setup_logging'), \
             patch('cli.EmbeddingClient'), \
             patch('cli.QdrantDB'), \
             patch('cli.LLMClient'):
            
            # Create CLI instance with mocked dependencies
            self.cli = RAGCLI('fake_config.yaml')
            
            # Mock documents configuration
            self.cli.documents_config = {'root_path': '/home/user/documents'}

    def test_empty_url(self):
        """Test handling of empty URLs."""
        self.assertEqual(self.cli._resolve_source_url(""), "")
        self.assertEqual(self.cli._resolve_source_url(None), None)

    def test_na_url(self):
        """Test handling of N/A URLs."""
        self.assertEqual(self.cli._resolve_source_url("N/A"), "N/A")

    def test_non_file_urls(self):
        """Test that non-file URLs are returned unchanged."""
        test_cases = [
            "https://example.com/doc.html",
            "http://localhost:8080/file.pdf",
            "ftp://server.com/file.txt",
            "mailto:user@example.com",
            "not-a-url-at-all"
        ]
        
        for url in test_cases:
            with self.subTest(url=url):
                self.assertEqual(self.cli._resolve_source_url(url), url)

    def test_absolute_file_urls(self):
        """Test that absolute file URLs are returned unchanged."""
        test_cases = [
            "file:/Users/user/documents/doc.pdf",
            "file:/home/user/docs/article.md",
            "file:C:\\Users\\user\\docs\\file.txt"  # Windows path
        ]
        
        for url in test_cases:
            with self.subTest(url=url):
                self.assertEqual(self.cli._resolve_source_url(url), url)

    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.expandvars')
    @patch('os.path.expanduser')
    def test_relative_file_urls(self, mock_expanduser, mock_expandvars, mock_join, mock_abspath):
        """Test resolution of relative file URLs."""
        # Setup mocks
        mock_expanduser.return_value = '/home/user/documents'
        mock_expandvars.return_value = '/home/user/documents'
        mock_join.return_value = '/home/user/documents/articles/doc.pdf'
        mock_abspath.return_value = '/home/user/documents/articles/doc.pdf'
        
        result = self.cli._resolve_source_url("file:articles/doc.pdf")
        
        # Verify the path resolution chain
        mock_expanduser.assert_called_once_with('/home/user/documents')
        mock_expandvars.assert_called_once_with('/home/user/documents')
        mock_join.assert_called_once_with('/home/user/documents', 'articles/doc.pdf')
        mock_abspath.assert_called_once_with('/home/user/documents/articles/doc.pdf')
        
        # Check final result
        self.assertEqual(result, "file:/home/user/documents/articles/doc.pdf")

    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.expandvars')
    @patch('os.path.expanduser')
    def test_nested_relative_file_urls(self, mock_expanduser, mock_expandvars, mock_join, mock_abspath):
        """Test resolution of nested relative file URLs."""
        # Setup mocks
        mock_expanduser.return_value = '/home/user/documents'
        mock_expandvars.return_value = '/home/user/documents'
        mock_join.return_value = '/home/user/documents/subfolder/another/doc.md'
        mock_abspath.return_value = '/home/user/documents/subfolder/another/doc.md'
        
        result = self.cli._resolve_source_url("file:subfolder/another/doc.md")
        
        # Check final result
        self.assertEqual(result, "file:/home/user/documents/subfolder/another/doc.md")

    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.expandvars')
    @patch('os.path.expanduser')
    def test_environment_variable_expansion(self, mock_expanduser, mock_expandvars, mock_join, mock_abspath):
        """Test that environment variables in document root are expanded."""
        # Setup CLI with env var in document root
        self.cli.documents_config = {'root_path': '$HOME/documents'}
        
        # Setup mocks
        mock_expanduser.return_value = '$HOME/documents'  # expanduser doesn't change this
        mock_expandvars.return_value = '/home/user/documents'  # expandvars resolves $HOME
        mock_join.return_value = '/home/user/documents/file.txt'
        mock_abspath.return_value = '/home/user/documents/file.txt'
        
        result = self.cli._resolve_source_url("file:file.txt")
        
        # Verify expansion chain
        mock_expanduser.assert_called_once_with('$HOME/documents')
        mock_expandvars.assert_called_once_with('$HOME/documents')
        
        self.assertEqual(result, "file:/home/user/documents/file.txt")

    @patch('os.path.abspath')
    @patch('os.path.join')
    @patch('os.path.expandvars')
    @patch('os.path.expanduser')
    def test_tilde_expansion(self, mock_expanduser, mock_expandvars, mock_join, mock_abspath):
        """Test that tilde in document root is expanded."""
        # Setup CLI with tilde in document root
        self.cli.documents_config = {'root_path': '~/documents'}
        
        # Setup mocks
        mock_expanduser.return_value = '/home/user/documents'  # expanduser resolves ~
        mock_expandvars.return_value = '/home/user/documents'
        mock_join.return_value = '/home/user/documents/file.txt'
        mock_abspath.return_value = '/home/user/documents/file.txt'
        
        result = self.cli._resolve_source_url("file:file.txt")
        
        # Verify expansion chain
        mock_expanduser.assert_called_once_with('~/documents')
        mock_expandvars.assert_called_once_with('/home/user/documents')
        
        self.assertEqual(result, "file:/home/user/documents/file.txt")

    def test_default_document_root(self):
        """Test behavior when document root is not configured."""
        # Remove document root config
        self.cli.documents_config = {}
        
        with patch('os.path.abspath') as mock_abspath, \
             patch('os.path.join') as mock_join, \
             patch('os.path.expandvars') as mock_expandvars, \
             patch('os.path.expanduser') as mock_expanduser:
            
            # Setup mocks
            mock_expanduser.return_value = './documents'
            mock_expandvars.return_value = './documents'
            mock_join.return_value = './documents/file.txt'
            mock_abspath.return_value = '/current/dir/documents/file.txt'
            
            result = self.cli._resolve_source_url("file:file.txt")
            
            # Should use default './documents'
            mock_expanduser.assert_called_once_with('./documents')
            mock_join.assert_called_once_with('./documents', 'file.txt')
            
            self.assertEqual(result, "file:/current/dir/documents/file.txt")


class TestURLResolutionIntegration(unittest.TestCase):
    """Integration tests for URL resolution with real file system operations."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('cli.ConfigManager'), \
             patch('cli.setup_logging'), \
             patch('cli.EmbeddingClient'), \
             patch('cli.QdrantDB'), \
             patch('cli.LLMClient'):
            
            self.cli = RAGCLI('fake_config.yaml')

    def test_real_path_resolution(self):
        """Test URL resolution with real path operations."""
        # Use a real temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            self.cli.documents_config = {'root_path': temp_dir}
            
            result = self.cli._resolve_source_url("file:test/document.pdf")
            
            # Should create absolute path
            expected = f"file:{os.path.abspath(os.path.join(temp_dir, 'test/document.pdf'))}"
            self.assertEqual(result, expected)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        self.cli.documents_config = {'root_path': '/tmp'}
        
        # Empty file path after 'file:'
        result = self.cli._resolve_source_url("file:")
        expected = f"file:{os.path.abspath('/tmp')}"
        self.assertEqual(result, expected)
        
        # Just 'file:' with slash
        result = self.cli._resolve_source_url("file:/")
        expected = "file:/"  # Already absolute
        self.assertEqual(result, expected)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)