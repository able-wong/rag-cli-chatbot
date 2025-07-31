import logging
import logging.handlers
import sys
import os
from typing import Dict, Any, Optional

def _parse_file_size(size_str: str) -> int:
    """Parse file size string (e.g., '10MB', '1GB') to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes if no unit specified
        return int(size_str)

def _create_log_handler(config: Dict[str, Any]) -> Optional[logging.Handler]:
    """Create appropriate log handler based on configuration."""
    output_type = config.get('output', 'stderr').lower()
    
    if output_type == 'none':
        # Create a null handler that discards all log messages
        return logging.NullHandler()
    
    elif output_type == 'console':
        # Output to stdout (mixes with chat interface)
        return logging.StreamHandler(sys.stdout)
    
    elif output_type == 'stderr':
        # Output to stderr (separates from chat interface)
        return logging.StreamHandler(sys.stderr)
    
    elif output_type == 'file':
        # Output to rotating file
        file_config = config.get('file', {})
        log_path = file_config.get('path', 'logs/rag-cli.log')
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Parse file size and backup count
        max_bytes = _parse_file_size(file_config.get('max_size', '10MB'))
        backup_count = file_config.get('backup_count', 5)
        
        return logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    
    else:
        # Default to stderr for unknown output types
        return logging.StreamHandler(sys.stderr)

def setup_logging(config: Dict[str, Any] = None) -> None:
    """Set up logging configuration with flexible output options."""
    if config is None:
        config = {}
    
    # Get log level
    log_level = config.get('level', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    
    # Get log format
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Clear any existing handlers on the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and configure the appropriate handler
    handler = _create_log_handler(config)
    if handler:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(level)
    
    # Set specific logger levels if needed
    loggers_config = config.get('loggers', {})
    for logger_name, logger_config in loggers_config.items():
        logger = logging.getLogger(logger_name)
        if 'level' in logger_config:
            logger.setLevel(getattr(logging, logger_config['level'].upper(), logging.INFO))