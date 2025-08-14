import logging
import os
from typing import List, Any, Dict
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config_manager import ConfigManager
from embedding_client import EmbeddingClient
from qdrant_db import QdrantDB
from llm_client import LLMClient
from query_rewriter import QueryRewriter
from search_service import SearchService
from logging_config import setup_logging

logger = logging.getLogger(__name__)

# Display constants
DISPLAY_TEXT_TRUNCATE_LENGTH = 160  # Characters to show before truncating with "..."

class RAGCLI:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.console = Console()
        self.config_manager = ConfigManager(config_path)
        
        # Setup logging
        setup_logging(self.config_manager.get_logging_config())
        
        # Initialize components
        self.embedding_client = None
        self.qdrant_db = None
        self.llm_client = None
        self.query_rewriter = None
        self.search_service = None
        self.conversation_history = []
        self.last_rag_results = []
        
        # Load configuration
        self.rag_config = self.config_manager.get_rag_config()
        self.cli_config = self.config_manager.get('cli', {})
        self.documents_config = self.config_manager.get_documents_config()
        
        self.trigger_phrase = self.rag_config.get('trigger_phrase', '@knowledgebase')
        self.min_score = self.rag_config.get('min_score', 0.7)
        self.top_k = self.rag_config.get('top_k', 5)
        self.max_context_length = self.rag_config.get('max_context_length', 8000)
        self.max_history_length = self.cli_config.get('max_history_length', 20)
        self.use_hybrid_search = self.rag_config.get('use_hybrid_search', False)
        
        self._initialize_clients()
        self._initialize_conversation()
    
    def _initialize_clients(self):
        """Initialize all service clients."""
        try:
            # Initialize embedding client
            embedding_config = self.config_manager.get_embedding_config()
            self.embedding_client = EmbeddingClient(embedding_config)
            logger.info("Embedding client initialized")
            
            # Initialize Qdrant client
            vector_db_config = self.config_manager.get_vector_db_config()
            self.qdrant_db = QdrantDB(vector_db_config)
            logger.info("Qdrant client initialized")
            
            # Initialize LLM client
            llm_config = self.config_manager.get_llm_config()
            self.llm_client = LLMClient(llm_config)
            logger.info("LLM client initialized")
            
            # Initialize QueryRewriter with merged configs
            query_rewriter_config = self.config_manager.get('query_rewriter', {})
            rag_config = self.config_manager.get('rag', {})
            
            # Merge configs: query_rewriter settings + rag settings for QueryRewriter
            merged_config = query_rewriter_config.copy()
            merged_config['trigger_phrase'] = self.trigger_phrase
            merged_config['retrieval_strategy'] = rag_config.get('retrieval_strategy', 'rewrite')
            
            self.query_rewriter = QueryRewriter(self.llm_client, merged_config)
            logger.info("QueryRewriter initialized")
            
            # Initialize SearchService with optional soft filtering configuration
            search_service_config = self.config_manager.get('search_service', {})
            self.search_service = SearchService(self.qdrant_db, search_service_config)
            logger.info("SearchService initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def _initialize_conversation(self):
        """Initialize conversation with system prompt."""
        # Use configurable system prompt from config
        system_prompt = self.cli_config.get('system_prompt', 
            "You are a helpful AI assistant. Follow the task instructions carefully and use the specified context source as directed. "
            "If you don't know the answer based on the specified context or from conversation history, you can say you don't know.")
        
        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        logger.info("Conversation initialized with configurable system prompt")
    
    def _get_help_text(self) -> str:
        """Return the help text string."""
        base_help = ("🤖 Welcome to RAG CLI Chatbot!\n\n"
                    "Commands:\n"
                    "- Type normally for general chat\n"
                    "- Use @knowledgebase to search the knowledge base\n"
                    "- /clear - Clear conversation history\n"
                    "- /help - Display this help message\n"
                    "- /info - Display system configuration\n"
                    "- /bye - Exit the chatbot\n"
                    "- /doc <number> - View detailed document content")
        
        # Add query pattern examples if hybrid search enabled
        if self.use_hybrid_search:
            query_examples = ("\n\nQuery Patterns:\n\n"
                            "📋 Search Mode (document discovery):\n"
                            "- 'search @knowledgebase on machine learning'\n"
                            "- '@knowledgebase find papers by Smith from 2024'\n"
                            "- 'get @knowledgebase articles about Python'\n\n"
                            
                            "🔍 Search + Action (find then analyze):\n"
                            "- 'search @knowledgebase on AI, explain the key concepts'\n"
                            "- '@knowledgebase find papers on neural networks and summarize'\n"
                            "- 'get @knowledgebase docs on Python, what are the benefits'\n\n"
                            
                            "💬 Direct Questions (knowledge consultation):\n"
                            "- '@knowledgebase what is machine learning'\n"
                            "- '@knowledgebase explain the benefits of Python'\n"
                            "- '@knowledgebase compare REST vs GraphQL APIs'\n\n"
                            
                            "🏷️ Filter Examples:\n"
                            "- Author: 'papers by Smith', 'research by Dr. Johnson'\n"
                            "- Date: 'from 2024', 'articles since 2023', 'before 2025'\n"
                            "- Tags: 'about Python', 'tagged as AI', 'on neural networks'\n"
                            "- Exclusions: 'not by Smith', 'without tag gemini', 'excluding robotics'")
            base_help += query_examples
        
        return base_help


    def _format_three_filter_display(self, query_analysis: Dict[str, Any]) -> str:
        """Format three-filter display with clear type indicators."""
        filter_parts = []
        
        # Hard filters (must match) - show with 🔒
        hard_filters = query_analysis.get('hard_filters', {})
        if hard_filters:
            hard_parts = []
            for key, value in hard_filters.items():
                if value:
                    hard_parts.append(self._format_single_filter(key, value))
            if hard_parts:
                filter_parts.append(f"🔒 {', '.join(hard_parts)}")
        
        # Negation filters (must not match) - show with 🚫  
        negation_filters = query_analysis.get('negation_filters', {})
        if negation_filters:
            neg_parts = []
            for key, value in negation_filters.items():
                if value:
                    neg_parts.append(f"NOT {self._format_single_filter(key, value)}")
            if neg_parts:
                filter_parts.append(f"🚫 {', '.join(neg_parts)}")
        
        # Soft filters (boost if match) - show with ⭐
        soft_filters = query_analysis.get('soft_filters', {})
        if soft_filters:
            soft_parts = []
            for key, value in soft_filters.items():
                if value:
                    soft_parts.append(self._format_single_filter(key, value))
            if soft_parts:
                filter_parts.append(f"⭐ {', '.join(soft_parts)}")
        
        return "; ".join(filter_parts)
    
    def _format_single_filter(self, key: str, value: Any) -> str:
        """Format a single filter key-value pair."""
        if key == "author":
            return f"author: {value}"
        elif key == "tags":
            if isinstance(value, list):
                tags_str = ", ".join(str(tag) for tag in value)
                return f"tags: [{tags_str}]"
            else:
                return f"tags: {value}"
        elif key == "publication_date":
            if isinstance(value, dict):
                # Handle date range objects
                if "gte" in value and "lt" in value:
                    start_date = value["gte"][:7] if len(value["gte"]) > 7 else value["gte"]
                    end_date = value["lt"][:7] if len(value["lt"]) > 7 else value["lt"]
                    return f"date: {start_date} to {end_date}"
                elif "gte" in value:
                    start_date = value["gte"][:7] if len(value["gte"]) > 7 else value["gte"]
                    return f"date: from {start_date}"
                elif "lt" in value:
                    end_date = value["lt"][:7] if len(value["lt"]) > 7 else value["lt"]
                    return f"date: before {end_date}"
            else:
                return f"date: {value}"
        else:
            return f"{key}: {value}"

    def _display_panel_message(self, message: str, title: str, border_style: str = "blue"):
        """Helper method to display a message within a rich.panel.Panel."""
        panel = Panel(
            message,
            title=title,
            border_style=border_style
        )
        self.console.print(panel)
        self.console.print()

    def _display_welcome(self):
        """Display welcome message."""
        self._display_panel_message(self._get_help_text(), "RAG CLI Chatbot")
    
    def _display_help(self):
        """Display help message."""
        self._display_panel_message(self._get_help_text(), "RAG CLI Chatbot Help")

    def _display_info(self):
        """Display system configuration information."""
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Category", style="bold blue", width=20)
        table.add_column("Setting", style="bold cyan", width=25)
        table.add_column("Value", style="white")

        # LLM Info
        llm_provider = self.config_manager.get('llm.provider', 'N/A')
        llm_model = self.config_manager.get(f'llm.{llm_provider}.model', self.config_manager.get('llm.model', 'N/A'))
        
        table.add_row("LLM", "Provider", llm_provider)
        table.add_row("", "Model", llm_model)
        
        # Embedding Info
        embedding_provider = self.config_manager.get('embedding.provider', 'N/A')
        embedding_model = self.config_manager.get(f'embedding.{embedding_provider}.model', self.config_manager.get('embedding.model', 'N/A'))
        
        table.add_row("Embedding", "Provider", embedding_provider)
        table.add_row("", "Model", embedding_model)
        
        # Vector DB Info
        vector_db_provider = self.config_manager.get('vector_db.provider', 'N/A')
        table.add_row("Vector DB", "Provider", vector_db_provider)
        
        if self.config_manager.get('vector_db.url'):
            db_location = self.config_manager.get('vector_db.url')
            table.add_row("", "URL", db_location)
        else:
            db_location = f"{self.config_manager.get('vector_db.host', 'N/A')}:{self.config_manager.get('vector_db.port', 'N/A')}"
            table.add_row("", "Location", db_location)
        
        collection_name = self.config_manager.get('vector_db.collection_name', 'N/A')
        table.add_row("", "Collection", collection_name)

        # RAG settings
        retrieval_strategy = self.config_manager.get('rag.retrieval_strategy', 'rewrite')
        hybrid_search_enabled = self.config_manager.get('rag.use_hybrid_search', False)
        table.add_row("RAG", "retrieval_strategy", retrieval_strategy)
        table.add_row("", "hybrid_search", str(hybrid_search_enabled))
        table.add_row("", "top_k", str(self.config_manager.get('rag.top_k', 'N/A')))
        table.add_row("", "min_score", str(self.config_manager.get('rag.min_score', 'N/A')))
        
        # Show system prompt status
        system_prompt = self.config_manager.get('rag.system_prompt', '')
        if system_prompt and system_prompt.strip():
            prompt_preview = system_prompt.strip()[:50] + "..." if len(system_prompt.strip()) > 50 else system_prompt.strip()
            table.add_row("", "system_prompt", prompt_preview)
        else:
            table.add_row("", "system_prompt", "configured")

        # Logging Info
        logging_output = self.config_manager.get('logging.output', 'N/A')
        table.add_row("Logging", "Output", logging_output)
        if logging_output == 'file':
            log_path = self.config_manager.get('logging.file.path', 'N/A')
            table.add_row("", "Log Path", log_path)

        # Documents Info
        doc_root = self.config_manager.get('documents.root_path', 'N/A')
        table.add_row("Documents", "Root Path", doc_root)

        panel = Panel(
            table,
            title="System Configuration",
            border_style="green",
            expand=False
        )
        self.console.print(panel)
    
    def _analyze_and_transform_query(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze and transform user query using QueryRewriter.
        
        Returns:
            Dict containing:
            - search_rag: boolean indicating if RAG search should be performed
            - embedding_source_text: optimized text for vector search
            - llm_query: refined prompt for LLM generation
        """
        try:
            # Check if query rewriter is enabled
            query_rewriter_config = self.config_manager.get('query_rewriter', {})
            if not query_rewriter_config.get('enabled', True):
                # Fallback to simple trigger detection if disabled
                return self._create_simple_query_analysis(user_input)
            
            # Use QueryRewriter for analysis
            result = self.query_rewriter.transform_query(user_input)
            logger.info(f"Query transformed. RAG: {result['search_rag']}, Embedding: '{result['embedding_source_text'][:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback to simple analysis
            return self._create_simple_query_analysis(user_input)
    
    def _create_simple_query_analysis(self, user_input: str) -> Dict[str, Any]:
        """Create simple query analysis as fallback when QueryRewriter fails."""
        has_trigger = self.trigger_phrase.lower() in user_input.lower()
        clean_query = user_input.replace(self.trigger_phrase, '').strip() if has_trigger else user_input
        
        # Only trigger RAG if there's content after the trigger phrase.
        search_rag = has_trigger and bool(clean_query)
        
        return {
            'search_rag': search_rag,
            'embedding_source_text': clean_query,
            'llm_query': user_input,
            'hard_filters': {},
            'negation_filters': {},
            'soft_filters': {}
        }
    
    def _perform_rag_search(self, embedding_text: str, query_analysis: Dict[str, Any]) -> List[Any]:
        """Perform RAG search using SearchService with all filter types."""
        try:
            # Generate embedding for the optimized text
            query_embedding = self.embedding_client.get_embedding(embedding_text)
            
            # Extract all filter types from query analysis
            hard_filters = query_analysis.get('hard_filters', {})
            negation_filters = query_analysis.get('negation_filters', {})
            soft_filters = query_analysis.get('soft_filters', {})
            
            # Use SearchService unified search with all filter types
            results = self.search_service.unified_search(
                query_vector=query_embedding,
                top_k=self.top_k,
                hard_filters=hard_filters if self.use_hybrid_search else None,
                negation_filters=negation_filters if self.use_hybrid_search else None,
                soft_filters=soft_filters if self.use_hybrid_search else None,
                score_threshold=self.min_score
            )
            
            logger.info(f"RAG search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    def _should_use_rag_context(self, results: List[Any]) -> bool:
        """Determine if RAG context should be used based on results."""
        if not results:
            return False
        
        # Check if the best result meets minimum score threshold
        best_score = results[0].score if results else 0
        return best_score >= self.min_score
    
    def _build_rag_context(self, results: List[Any]) -> str:
        """Build context string from RAG search results."""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Get document content from payload - try multiple possible keys
            content = (result.payload.get('content', '') or 
                      result.payload.get('text', '') or 
                      result.payload.get('chunk_text', '') or
                      result.payload.get('original_text', ''))
            
            # Build rich source information
            title = result.payload.get('title', '')
            author = result.payload.get('author', '')
            pub_date = result.payload.get('publication_date', '')
            
            # Create source attribution
            source_parts = []
            if title:
                source_parts.append(title)
            if author:
                source_parts.append(f"by {author}")
            if pub_date:
                source_parts.append(f"({pub_date})")
            
            source = " ".join(source_parts) if source_parts else f"Document {i}"
            
            context_part = f"Source {i} - {source}:\n{content}\n"
            
            # Check if adding this would exceed max context length
            if current_length + len(context_part) > self.max_context_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def _build_prompt_with_context(self, llm_query: str, context: str) -> str:
        """Build user prompt with RAG context using structured LLM query."""
        return f"""Context from knowledge base:
{context}

Task: {llm_query}

If the context doesn't contain enough information to complete the task, please state that clearly."""
    
    def _build_no_answer_prompt(self, llm_query: str) -> str:
        """Build prompt for when no relevant context is found."""
        # Extract the core question from the LLM query for user feedback
        # Remove common instruction words to get the essence of what they asked
        clean_query = llm_query.lower()
        for phrase in ['based on the provided context', 'explain', 'describe', 'list', 'compare']:
            clean_query = clean_query.replace(phrase, '').strip()
        clean_query = clean_query.strip('.,!?').strip()
        
        return f"""I searched the knowledge base for information related to your query about \"{clean_query}\", but I couldn't find relevant information to answer your question. The available documents don't seem to contain information that matches your query with sufficient confidence.
Is there anything else I can help you with, or would you like to rephrase your question?"""
    
    def _manage_conversation_history(self):
        """Manage conversation history length."""
        # Keep system prompt + last N turns (user + assistant pairs)
        if len(self.conversation_history) > (self.max_history_length * 2 + 1):
            # Keep system prompt and remove oldest user-assistant pairs
            system_prompt = self.conversation_history[0]
            recent_history = self.conversation_history[-(self.max_history_length * 2):]
            self.conversation_history = [system_prompt] + recent_history
            logger.info("Conversation history trimmed")
    
    def _display_rag_results(self, results: List[Any]):
        """Display RAG search results to user."""
        if not results or not self.cli_config.get('show_rag_results', True):
            return
        
        self.console.print("\n📚 [blue]Retrieved from knowledge base:[/blue]")
        
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", min_width=20)
        table.add_column("Author", style="yellow", min_width=12, max_width=20)
        table.add_column("Date", style="magenta", width=12)
        table.add_column("Tags", style="bright_blue", max_width=25)
        
        if self.cli_config.get('show_scores', True):
            table.add_column("Score", style="green", width=8)
        
        for i, result in enumerate(results, 1):
            # Extract enhanced metadata with null handling
            title = result.payload.get('title') or 'Untitled'
            author = result.payload.get('author') or 'Unknown'
            pub_date = result.payload.get('publication_date') or 'N/A'
            tags = result.payload.get('tags') or []
            
            # Format tags display
            if tags and isinstance(tags, list) and len(tags) > 0:
                tags_display = ", ".join(str(tag) for tag in tags[:3] if tag)  # Show first 3 tags
                if len(tags) > 3:
                    tags_display += f" +{len(tags)-3}"
            else:
                tags_display = "None"
            
            # Truncate long titles
            if len(title) > 25:
                title = title[:22] + "..."
            
            # Truncate long author names
            if len(author) > 18:
                author = author[:15] + "..."
            
            row_data = [str(i), title, author, pub_date, tags_display]
            
            if self.cli_config.get('show_scores', True):
                row_data.append(f"{result.score:.3f}")
            
            table.add_row(*row_data)
        
        self.console.print(table)
        self.console.print("💡 [dim]Use /doc <number> to see full content and metadata[/dim]\n")
    
    def _resolve_source_url(self, source_url: str) -> str:
        """Resolve source URL to absolute path for clickable terminal links."""
        if not source_url or source_url == 'N/A':
            return source_url
        
        # Handle file: URLs
        if source_url.startswith('file:'):
            # Extract the path part (remove 'file:' prefix)
            file_path = source_url[5:]  # Remove 'file:' prefix
            
            # Check if it's already an absolute path
            if os.path.isabs(file_path):
                # Already absolute, return as-is
                return source_url
            else:
                # Relative path, resolve using document root
                doc_root = self.documents_config.get('root_path', './documents')
                
                # Expand ~ and environment variables in doc_root if present
                doc_root = os.path.expanduser(os.path.expandvars(doc_root))
                
                # Join with document root and convert to absolute path
                absolute_path = os.path.abspath(os.path.join(doc_root, file_path))
                
                # Return as file: URL with absolute path
                return f"file:{absolute_path}"
        
        # For non-file URLs (http, https, etc.), return as-is
        return source_url
    
    def _display_document_details(self, doc_index: int):
        """Display detailed document content."""
        if not self.last_rag_results or doc_index < 1 or doc_index > len(self.last_rag_results):
            self.console.print("❌ [red]Invalid document number[/red]")
            return
        
        result = self.last_rag_results[doc_index - 1]
        
        # Get content
        content = (result.payload.get('content', '') or 
                  result.payload.get('text', '') or 
                  result.payload.get('chunk_text', '') or
                  result.payload.get('original_text', ''))
        
        # Get metadata with null handling
        title = result.payload.get('title') or 'Untitled Document'
        author = result.payload.get('author') or 'Unknown Author'
        pub_date = result.payload.get('publication_date') or 'Unknown Date'
        tags = result.payload.get('tags') or []
        source_url = result.payload.get('source_url') or 'N/A'
        chunk_index = result.payload.get('chunk_index', 'N/A')
        file_size = result.payload.get('file_size', 'N/A')
        
        # Create metadata table
        metadata_table = Table(show_header=False, box=None, padding=(0, 1))
        metadata_table.add_column("Field", style="bold cyan", width=15)
        metadata_table.add_column("Value", style="white")
        
        metadata_table.add_row("Title:", title)
        metadata_table.add_row("Author:", author)
        metadata_table.add_row("Date:", pub_date)
        metadata_table.add_row("Score:", f"{result.score:.3f}")
        metadata_table.add_row("Chunk:", str(chunk_index))
        
        if tags and isinstance(tags, list) and len(tags) > 0:
            tags_str = ", ".join(str(tag) for tag in tags if tag)
            metadata_table.add_row("Tags:", tags_str)
        
        if source_url != 'N/A':
            # Truncate long URLs for display
            display_url = source_url if len(source_url) <= 50 else source_url[:47] + "..."
            metadata_table.add_row("Source:", display_url)
        
        if file_size != 'N/A' and isinstance(file_size, (int, str)):
            try:
                size_bytes = int(file_size)
                if size_bytes > 1024 * 1024:
                    size_display = f"{size_bytes / (1024 * 1024):.1f} MB"
                elif size_bytes > 1024:
                    size_display = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_display = f"{size_bytes} bytes"
                metadata_table.add_row("File Size:", size_display)
            except Exception:
                pass
        
        # Create simple metadata display
        metadata_lines = []
        metadata_lines.append(f"[bold cyan]Title:[/bold cyan] {title}")
        metadata_lines.append(f"[bold cyan]Author:[/bold cyan] {author}")
        metadata_lines.append(f"[bold cyan]Date:[/bold cyan] {pub_date}")
        metadata_lines.append(f"[bold cyan]Score:[/bold cyan] {result.score:.3f}")
        metadata_lines.append(f"[bold cyan]Chunk:[/bold cyan] {chunk_index}")
        
        if tags and isinstance(tags, list) and len(tags) > 0:
            tags_str = ", ".join(str(tag) for tag in tags if tag)
            metadata_lines.append(f"[bold cyan]Tags:[/bold cyan] {tags_str}")
        
        metadata_section = "\n".join(metadata_lines)
        content_display = f"{metadata_section}\n\n[bold]Content:[/bold]\n{content}"
        
        panel = Panel(
            content_display,
            title=f"📄 Document {doc_index} Details",
            border_style="cyan",
            expand=False
        )
        self.console.print(panel)
        
        # Display clickable source URL after the box
        if source_url != 'N/A':
            resolved_url = self._resolve_source_url(source_url)
            # Use print() directly to avoid Rich's word wrapping that breaks URLs
            print(f"Source: {resolved_url}")
    
    def _handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        user_input = user_input.strip()
        
        if user_input == '/bye':
            self.console.print("👋 [yellow]Goodbye![/yellow]")
            return True
        
        elif user_input == '/clear':
            self._initialize_conversation()
            self.console.print("🧹 [green]Conversation history cleared[/green]")
            return False
        
        elif user_input == '/help':
            self._display_help()
            return False
        
        elif user_input == '/info':
            self._display_info()
            return False
        
        elif user_input.startswith('/doc '):
            try:
                doc_num = int(user_input.split()[1])
                self._display_document_details(doc_num)
            except (IndexError, ValueError):
                self.console.print("❌ [red]Usage: /doc <number>[/red]")
            return False
        
        return False
    
    def chat(self):
        """Main chat loop."""
        self._display_welcome()
        
        while True:
            try:
                # Use standard input with ANSI color codes for better terminal compatibility
                try:
                    user_input = input("\n\033[1;34mYou:\033[0m ").strip()
                except EOFError:
                    # Handle Ctrl+D gracefully
                    self.console.print("\n👋 [yellow]Goodbye![/yellow]")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C gracefully - exit the application
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        break  # Exit if /bye command
                    continue
                
                # Analyze and transform query
                query_analysis = self._analyze_and_transform_query(user_input)
                use_rag = query_analysis['search_rag']
                
                if use_rag:
                    # Select appropriate embedding text based on retrieval strategy
                    retrieval_strategy = self.config_manager.get('rag.retrieval_strategy', 'rewrite')
                    embedding_texts = query_analysis.get('embedding_texts', {})
                    
                    if retrieval_strategy == 'hyde' and 'hyde' in embedding_texts and embedding_texts['hyde']:
                        # Use first hyde text for backward compatibility
                        embedding_text = embedding_texts['hyde'][0]
                    elif 'rewrite' in embedding_texts:
                        embedding_text = embedding_texts['rewrite']
                    else:
                        # Fallback to legacy field for backward compatibility
                        embedding_text = query_analysis.get('embedding_source_text', '')
                    
                    search_display = embedding_text[:DISPLAY_TEXT_TRUNCATE_LENGTH] + "..." if len(embedding_text) > DISPLAY_TEXT_TRUNCATE_LENGTH else embedding_text
                    
                    if self.use_hybrid_search:
                        filter_info = self._format_three_filter_display(query_analysis)
                        if filter_info:
                            self.console.print(f"🔍 [dim]Searching knowledge base with '{search_display}' ({filter_info})...[/dim]")
                        else:
                            self.console.print(f"🔍 [dim]Searching knowledge base with '{search_display}'...[/dim]")
                    else:
                        self.console.print(f"🔍 [dim]Searching knowledge base with '{search_display}'...[/dim]")
                    
                    rag_results = self._perform_rag_search(embedding_text, query_analysis)
                    self.last_rag_results = rag_results
                    
                    if self._should_use_rag_context(rag_results):
                        # Use RAG context with structured LLM query
                        context = self._build_rag_context(rag_results)
                        prompt = self._build_prompt_with_context(query_analysis['llm_query'], context)
                    else:
                        # No relevant context found
                        prompt = self._build_no_answer_prompt(query_analysis['llm_query'])
                        rag_results = []  # Clear results since they're not relevant
                else:
                    # Regular chat without RAG using structured LLM query
                    prompt = query_analysis['llm_query']
                    rag_results = []
                
                # Add user message to history
                self.conversation_history.append({"role": "user", "content": prompt})
                
                # Get LLM response
                llm_prompt_display = query_analysis['llm_query'][:DISPLAY_TEXT_TRUNCATE_LENGTH] + "..." if len(query_analysis['llm_query']) > DISPLAY_TEXT_TRUNCATE_LENGTH else query_analysis['llm_query']
                self.console.print(f"🤖 [dim]Thinking with LLM prompt '{llm_prompt_display}'...[/dim]")
                try:
                    response = self.llm_client.get_llm_response(self.conversation_history)
                    
                    # Display response
                    self.console.print("\n[bold green]Assistant:[/bold green]")
                    self.console.print(Markdown(response))
                    
                    # Add assistant response to history
                    self.conversation_history.append({"role": "assistant", "content": response})
                    
                    # Display RAG results if applicable
                    if rag_results:
                        self._display_rag_results(rag_results)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    self.console.print(f"❌ [red]{error_msg}[/red]")
                    logger.error(f"LLM call failed: {e}")
                
                # Manage conversation history length
                self._manage_conversation_history()
                
            except KeyboardInterrupt:
                self.console.print("\n👋 [yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"❌ [red]Unexpected error: {str(e)}[/red]")
                logger.error(f"Unexpected error in chat loop: {e}")

if __name__ == "__main__":
    try:
        cli = RAGCLI()
        cli.chat()
    except Exception as e:
        console = Console()
        console.print(f"❌ [red]Failed to start RAG CLI: {str(e)}[/red]")
        logger.error(f"Failed to start application: {e}")
        exit(1)