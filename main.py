#!/usr/bin/env python3
"""
RAG CLI Chatbot - Main Entry Point

A command-line chatbot with Retrieval-Augmented Generation (RAG) capabilities.
Supports knowledge base search using Qdrant vector database and various LLM providers.

Usage:
    python main.py [config_path]

Example:
    python main.py
    python main.py config/custom_config.yaml
"""

import sys
import os
import argparse
from rich.console import Console

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cli import RAGCLI

def main():
    """Main entry point for the RAG CLI Chatbot."""
    parser = argparse.ArgumentParser(
        description="RAG CLI Chatbot - Interactive chatbot with knowledge base search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Use default config
    python main.py config/custom_config.yaml   # Use custom config
    
Commands available in chat:
    /bye                 - Exit the chatbot
    /clear               - Clear conversation history  
    /doc <number>        - View detailed document content
    @knowledgebase       - Trigger knowledge base search
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='RAG CLI Chatbot 1.0.0'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output showing detailed query analysis and timing'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        console.print(f"❌ [red]Configuration file not found: {args.config}[/red]")
        console.print("💡 [yellow]Please create a config file or check the path[/yellow]")
        console.print("📝 [blue]Example config available at: config/config.yaml[/blue]")
        sys.exit(1)
    
    try:
        # Initialize and start the chatbot
        console.print("🚀 [green]Starting RAG CLI Chatbot...[/green]")
        console.print(f"📋 [dim]Using config: {args.config}[/dim]")
        
        cli = RAGCLI(args.config, verbose=args.verbose)
        cli.chat()
        
    except KeyboardInterrupt:
        console.print("\n\n👋 [yellow]Interrupted by user. Goodbye![/yellow]")
        sys.exit(0)
        
    except Exception as e:
        console.print(f"\n❌ [red]Failed to start RAG CLI Chatbot: {str(e)}[/red]")
        console.print("💡 [yellow]Please check your configuration and dependencies[/yellow]")
        sys.exit(1)

if __name__ == "__main__":
    main()