#!/usr/bin/env python3
"""
Unit tests for logging configuration functionality.
Tests the logging_config module with various configurations.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import logging
import logging.handlers
import sys
import os
import tempfile
import shutil

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logging_config import setup_logging, _parse_file_size, _create_log_handler


class TestFileSizeParsing(unittest.TestCase):
    """Test cases for file size parsing functionality."""

    def test_kilobyte_parsing(self):
        """Test parsing of KB values."""
        self.assertEqual(_parse_file_size("1KB"), 1024)
        self.assertEqual(_parse_file_size("10KB"), 10 * 1024)
        self.assertEqual(_parse_file_size("2.5KB"), int(2.5 * 1024))

    def test_megabyte_parsing(self):
        """Test parsing of MB values."""
        self.assertEqual(_parse_file_size("1MB"), 1024 * 1024)
        self.assertEqual(_parse_file_size("10MB"), 10 * 1024 * 1024)
        self.assertEqual(_parse_file_size("2.5MB"), int(2.5 * 1024 * 1024))

    def test_gigabyte_parsing(self):
        """Test parsing of GB values."""
        self.assertEqual(_parse_file_size("1GB"), 1024 * 1024 * 1024)
        self.assertEqual(_parse_file_size("2GB"), 2 * 1024 * 1024 * 1024)

    def test_bytes_parsing(self):
        """Test parsing of byte values without units."""
        self.assertEqual(_parse_file_size("5000"), 5000)
        self.assertEqual(_parse_file_size("1024"), 1024)

    def test_case_insensitive_parsing(self):
        """Test that parsing is case insensitive."""
        self.assertEqual(_parse_file_size("1kb"), 1024)
        self.assertEqual(_parse_file_size("1Mb"), 1024 * 1024)
        self.assertEqual(_parse_file_size("1gb"), 1024 * 1024 * 1024)

    def test_whitespace_handling(self):
        """Test that whitespace is handled properly."""
        self.assertEqual(_parse_file_size(" 1MB "), 1024 * 1024)
        self.assertEqual(_parse_file_size("  10KB  "), 10 * 1024)


class TestLogHandlerCreation(unittest.TestCase):
    """Test cases for log handler creation."""

    def test_null_handler_creation(self):
        """Test creation of null handler for 'none' output."""
        config = {'output': 'none'}
        handler = _create_log_handler(config)
        self.assertIsInstance(handler, logging.NullHandler)

    def test_console_handler_creation(self):
        """Test creation of console handler."""
        config = {'output': 'console'}
        handler = _create_log_handler(config)
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stdout)

    def test_stderr_handler_creation(self):
        """Test creation of stderr handler."""
        config = {'output': 'stderr'}
        handler = _create_log_handler(config)
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stderr)

    def test_default_handler_creation(self):
        """Test default handler creation (should be stderr)."""
        config = {}
        handler = _create_log_handler(config)
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stderr)

    def test_unknown_output_type(self):
        """Test that unknown output types default to stderr."""
        config = {'output': 'unknown_type'}
        handler = _create_log_handler(config)
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stderr)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.dirname')
    def test_file_handler_creation(self, mock_dirname, mock_exists, mock_makedirs):
        """Test creation of rotating file handler."""
        mock_dirname.return_value = 'logs'
        mock_exists.return_value = False
        
        config = {
            'output': 'file',
            'file': {
                'path': 'logs/test.log',
                'max_size': '5MB',
                'backup_count': 3
            }
        }
        
        handler = _create_log_handler(config)
        
        # Should create directory if it doesn't exist
        mock_makedirs.assert_called_once_with('logs', exist_ok=True)
        
        # Should be a RotatingFileHandler
        self.assertIsInstance(handler, logging.handlers.RotatingFileHandler)
        self.assertEqual(handler.maxBytes, 5 * 1024 * 1024)
        self.assertEqual(handler.backupCount, 3)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.dirname')
    def test_file_handler_with_defaults(self, mock_dirname, mock_exists, mock_makedirs):
        """Test file handler creation with default values."""
        mock_dirname.return_value = 'logs'
        mock_exists.return_value = True  # Directory exists
        
        config = {'output': 'file'}
        handler = _create_log_handler(config)
        
        # Should not create directory if it exists
        mock_makedirs.assert_not_called()
        
        # Should use defaults
        self.assertIsInstance(handler, logging.handlers.RotatingFileHandler)
        self.assertEqual(handler.maxBytes, 10 * 1024 * 1024)  # Default 10MB
        self.assertEqual(handler.backupCount, 5)  # Default 5 backups


class TestLoggingSetup(unittest.TestCase):
    """Test cases for complete logging setup."""

    def setUp(self):
        """Set up test fixtures."""
        # Store the original root logger state
        self.original_handlers = logging.getLogger().handlers[:]
        self.original_level = logging.getLogger().level

    def tearDown(self):
        """Clean up after tests."""
        # Restore original root logger state
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in self.original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(self.original_level)

    def test_basic_setup(self):
        """Test basic logging setup."""
        config = {
            'level': 'DEBUG',
            'output': 'stderr',
            'format': '%(levelname)s: %(message)s'
        }
        
        setup_logging(config)
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)
        self.assertEqual(len(root_logger.handlers), 1)
        
        handler = root_logger.handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.level, logging.DEBUG)

    def test_default_configuration(self):
        """Test setup with no configuration (should use defaults)."""
        setup_logging()
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)
        self.assertEqual(len(root_logger.handlers), 1)
        
        handler = root_logger.handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stderr)

    def test_invalid_log_level(self):
        """Test handling of invalid log levels."""
        config = {'level': 'INVALID_LEVEL'}
        setup_logging(config)
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)  # Should default to INFO

    def test_handler_replacement(self):
        """Test that existing handlers are replaced."""
        # Add a dummy handler first
        dummy_handler = logging.StreamHandler()
        logging.getLogger().addHandler(dummy_handler)
        initial_count = len(logging.getLogger().handlers)
        
        setup_logging({'output': 'stderr'})
        
        # Should have exactly one handler after setup
        self.assertEqual(len(logging.getLogger().handlers), 1)
        # Should not be the dummy handler
        self.assertNotIn(dummy_handler, logging.getLogger().handlers)

    def test_custom_formatter(self):
        """Test custom formatter configuration."""
        config = {
            'format': 'CUSTOM: %(message)s',
            'date_format': '%H:%M:%S'
        }
        
        setup_logging(config)
        
        handler = logging.getLogger().handlers[0]
        formatter = handler.formatter
        
        # Test the formatter by formatting a dummy record
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='test message', args=(), exc_info=None
        )
        
        formatted = formatter.format(record)
        self.assertTrue(formatted.startswith('CUSTOM: test message'))

    @patch('logging_config._create_log_handler')
    def test_specific_logger_configuration(self, mock_create_handler):
        """Test configuration of specific loggers."""
        mock_handler = Mock()
        mock_create_handler.return_value = mock_handler
        
        config = {
            'level': 'INFO',
            'loggers': {
                'test.logger': {'level': 'DEBUG'},
                'another.logger': {'level': 'WARNING'}
            }
        }
        
        setup_logging(config)
        
        # Check specific logger levels
        test_logger = logging.getLogger('test.logger')
        another_logger = logging.getLogger('another.logger')
        
        self.assertEqual(test_logger.level, logging.DEBUG)
        self.assertEqual(another_logger.level, logging.WARNING)


class TestLoggingIntegration(unittest.TestCase):
    """Integration tests for logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clear root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_file_logging_integration(self):
        """Test actual file logging with rotation."""
        config = {
            'level': 'INFO',
            'output': 'file',
            'file': {
                'path': self.log_file,
                'max_size': '1KB',  # Small size to test rotation
                'backup_count': 2
            },
            'format': '%(levelname)s: %(message)s'
        }
        
        setup_logging(config)
        
        # Log some messages
        logger = logging.getLogger('test')
        logger.info('Test message 1')
        logger.warning('Test message 2')
        logger.error('Test message 3')
        
        # Force handler flush
        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Check that log file was created and contains messages
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn('INFO: Test message 1', content)
            self.assertIn('WARNING: Test message 2', content)
            self.assertIn('ERROR: Test message 3', content)

    def test_console_output_integration(self):
        """Test console output integration."""
        config = {
            'level': 'DEBUG',
            'output': 'console',
            'format': '%(levelname)s: %(message)s'
        }
        
        with patch('sys.stdout') as mock_stdout:
            setup_logging(config)
            
            logger = logging.getLogger('test')
            logger.info('Test console message')
            
            # Force handler flush
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
            
            # Check that stdout was used
            self.assertTrue(mock_stdout.write.called)

    def test_null_handler_integration(self):
        """Test that null handler properly discards messages."""
        config = {'output': 'none'}
        setup_logging(config)
        
        logger = logging.getLogger('test')
        
        # This should not raise any exceptions or produce output
        logger.debug('This message should be discarded')
        logger.info('This message should also be discarded')
        logger.error('Even errors should be discarded')
        
        # Verify null handler is in use
        root_logger = logging.getLogger()
        self.assertEqual(len(root_logger.handlers), 1)
        self.assertIsInstance(root_logger.handlers[0], logging.NullHandler)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)