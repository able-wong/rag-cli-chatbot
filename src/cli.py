import logging
import os
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config_manager import ConfigManager
from embedding_client import EmbeddingClient
from qdrant_db import QdrantDB
from llm_client import LLMClient
from logging_config import setup_logging

logger = logging.getLogger(__name__)

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
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def _initialize_conversation(self):
        """Initialize conversation with system prompt."""
        system_prompt = self.rag_config.get('system_prompt', 
            "You are a helpful AI assistant with access to a knowledge base.")
        
        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        logger.info("Conversation initialized with system prompt")
    
    def _display_welcome(self):
        """Display welcome message."""
        welcome_msg = self.cli_config.get('welcome_message', 
            "🤖 Welcome to RAG CLI Chatbot!\n\nCommands:\n- Type normally for general chat\n- Use @knowledgebase to search the knowledge base\n- /clear - Clear conversation history\n- /bye - Exit the chatbot\n- /doc <number> - View detailed document content")
        
        panel = Panel(
            welcome_msg,
            title="RAG CLI Chatbot",
            border_style="blue"
        )
        self.console.print(panel)
        self.console.print()
    
    def _detect_rag_trigger(self, user_input: str) -> bool:
        """Check if user input contains RAG trigger phrase."""
        return self.trigger_phrase.lower() in user_input.lower()
    
    def _perform_rag_search(self, query: str) -> List[Any]:
        """Perform RAG search and return results."""
        try:
            # Remove trigger phrase from query for embedding
            clean_query = query.replace(self.trigger_phrase, '').strip()
            
            # Generate embedding for the query
            query_embedding = self.embedding_client.get_embedding(clean_query)
            
            # Search in Qdrant
            results = self.qdrant_db.search(
                query_vector=query_embedding,
                limit=self.top_k,
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
    
    def _build_prompt_with_rag(self, query: str, context: str) -> str:
        """Build user prompt with RAG context."""
        return f"""Based on the following context from the knowledge base, please answer the question:

Context:
{context}

Question: {query.replace(self.trigger_phrase, '').strip()}

Please provide a helpful answer based on the context above. If the context doesn't contain enough information to answer the question, please state that clearly."""
    
    def _build_no_answer_prompt(self, query: str) -> str:
        """Build prompt for when no relevant context is found."""
        return f"""I searched the knowledge base for information related to: "{query.replace(self.trigger_phrase, '').strip()}"

However, I couldn't find relevant information in the knowledge base to answer your question. The available documents don't seem to contain information that matches your query with sufficient confidence.

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
            except:
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
                
                # Check for RAG trigger
                use_rag = self._detect_rag_trigger(user_input)
                
                if use_rag:
                    # Perform RAG search
                    self.console.print("🔍 [dim]Searching knowledge base...[/dim]")
                    rag_results = self._perform_rag_search(user_input)
                    self.last_rag_results = rag_results
                    
                    if self._should_use_rag_context(rag_results):
                        # Use RAG context
                        context = self._build_rag_context(rag_results)
                        prompt = self._build_prompt_with_rag(user_input, context)
                    else:
                        # No relevant context found
                        prompt = self._build_no_answer_prompt(user_input)
                        rag_results = []  # Clear results since they're not relevant
                else:
                    # Regular chat without RAG
                    prompt = user_input
                    rag_results = []
                
                # Add user message to history
                self.conversation_history.append({"role": "user", "content": prompt})
                
                # Get LLM response
                self.console.print("🤖 [dim]Thinking...[/dim]")
                try:
                    response = self.llm_client.get_llm_response(self.conversation_history)
                    
                    # Display response
                    self.console.print(f"\n[bold green]Assistant:[/bold green]")
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