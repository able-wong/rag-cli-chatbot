# RAG CLI Chatbot

A command-line chatbot with Retrieval-Augmented Generation (RAG) capabilities. Supports knowledge base search using Qdrant vector database and various LLM providers.

## 🎯 Features

- **Interactive CLI Interface**: Rich terminal interface with colors and formatting
- **RAG Capabilities**: Search and retrieve information from knowledge base
- **Hybrid Search**: Natural language metadata filtering (author, date, tags)
- **Dual Retrieval Strategies**: Choose between keyword-based "rewrite" or semantic "HyDE" search
- **Conversational Context Awareness**: Intelligent follow-up handling without re-searching
- **LLM-Based Query Transformation**: Smart query analysis and routing for better results
- **Configurable System Prompts**: Customize assistant role and personality
- **Multiple LLM Providers**: Support for Ollama and Gemini
- **Multiple Embedding Providers**: Ollama, Gemini, and SentenceTransformers
- **Vector Database**: Qdrant integration for semantic search
- **Smart Fallback**: "No answer" response when confidence is low
- **Conversation Management**: Maintains chat history with configurable limits
- **Command System**: Built-in commands for enhanced user experience

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

1. **Clone the repository** (or use the existing project):
   ```bash
   cd rag-cli-chatbot
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**:
   ```bash
   # Copy sample configuration and customize
   cp config/config.sample.yaml config/config.yaml
   # Edit config/config.yaml with your settings
   # See configuration section below
   ```

5. **Run the chatbot**:
   ```bash
   python3 main.py
   ```

## ⚙️ Configuration

The application uses `config/config.yaml` for configuration. Key settings:

### LLM Provider Setup

**Ollama (Local)**:
```yaml
llm:
  provider: "ollama"
  model: "llama3.2"
  base_url: "http://localhost:11434"
```

**Gemini (Cloud)**:
```yaml
llm:
  provider: "gemini"
  gemini:
    api_key: "your-api-key"  # Or set GEMINI_API_KEY env var
    model: "gemini-1.5-flash"
```

### Embedding Provider Setup

**SentenceTransformers (Local, Default)**:
```yaml
embedding:
  provider: "sentence_transformers"
  sentence_transformers:
    model: "all-MiniLM-L6-v2"
    device: "cpu"
```

**Ollama (Local)**:
```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"
```

### Vector Database Setup

**Local Qdrant**:
```yaml
vector_db:
  provider: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "knowledge_base"
```

**Qdrant Cloud**:
```yaml
vector_db:
  provider: "qdrant"
  url: "https://your-cluster.qdrant.io:6333"
  api_key: "your-api-key"  # Or set QDRANT_API_KEY env var
```

### RAG Retrieval Strategy

The application supports two retrieval strategies for knowledge base search:

**Rewrite Strategy (Default - Fast)**:
```yaml
rag:
  retrieval_strategy: "rewrite"  # Extract focused keywords from user queries
  trigger_phrase: "@knowledgebase"
  top_k: 10
  min_score: 0.3
```

**HyDE Strategy (Semantic - Better Understanding)**:
```yaml
rag:
  retrieval_strategy: "hyde"  # Generate hypothetical documents for semantic search
  trigger_phrase: "@knowledgebase"
  top_k: 10
  min_score: 0.3
```

**Strategy Comparison:**

| Strategy | Speed | Best For | Search Method |
|----------|-------|----------|---------------|
| **Rewrite** | Fast | Factual queries, keyword matching | `"neural networks training"` |
| **HyDE** | Slower | Conceptual questions, semantic understanding | `"Neural networks learn through backpropagation..."` |

**When to Use:**
- **Rewrite**: Start here for general use, good performance
- **HyDE**: Switch if you need better retrieval of conceptually related content

### Hybrid Search (Metadata Filtering)

**NEW FEATURE**: Hybrid search combines semantic similarity with metadata filtering for more precise results.

**Enable Hybrid Search**:
```yaml
rag:
  use_hybrid_search: true  # Enable natural language metadata filtering
  retrieval_strategy: "rewrite"  # Works with both rewrite and hyde
  trigger_phrase: "@knowledgebase"
```

**Requirements:**
- Qdrant collection must have payload indexes for: `tags`, `author`, `publication_date`
- These indexes are created during document ingestion
- System logs warnings if indexes are missing (performance impact)

**Natural Language Examples:**

| User Query | Extracted Filters | Result |
|------------|------------------|---------|
| `@knowledgebase papers by John Smith` | `{"author": "John Smith"}` | Only documents by John Smith |
| `@knowledgebase articles from 2023` | `{"publication_date": "2023"}` | Only 2023 publications |
| `@knowledgebase documents about Python` | `{"tags": ["python"]}` | Only Python-tagged documents |
| `@knowledgebase Smith's AI papers from 2023` | `{"author": "Smith", "tags": ["ai"], "publication_date": "2023"}` | Combined filtering |
| `@knowledgebase search on vibe coding from John Wong published in March 2025` | `{"author": "John Wong", "tags": ["vibe coding"], "publication_date": "2025-03"}` | Multi-attribute precision |

**Benefits:**
- **Precision**: Combine semantic search with structured filtering
- **Natural Interface**: No complex filter syntax required  
- **Flexible**: Works with both retrieval strategies
- **Performance**: Uses Qdrant indexes for efficient filtering

**Performance Notes:**
- Individual indexes only (Qdrant doesn't support compound indexes)
- Filter intersection handled by Qdrant efficiently
- Warning logs if required indexes are missing

### CLI Settings

**System Prompt Customization**:
```yaml
cli:
  system_prompt: "You are a helpful AI assistant. Follow the task instructions carefully and use the specified context source as directed. If you don't know the answer based on the specified context or from conversation history, you can say you don't know."
  max_history_length: 20  # Maximum conversation turns to keep
```

**Custom Role Example**:
```yaml
cli:
  system_prompt: "You are John's writing assistant. Your job is helping John research and write blog posts. Follow the task instructions carefully and use the specified context source as directed. If you don't know the answer based on the specified context or from conversation history, you can say you don't know."
```

### Qdrant collection "documents" Schema Keys:

- **chunk_text**: The actual text content of the chunk
- **original_text**: Original unprocessed document text
- **chunk_index**: Sequential index of this chunk within the document
- **source_url**: Source URL with protocol (e.g., file:documents/article.pdf, https://example.com/doc.html)
- **file_extension**: File extension (e.g., .pdf, .html, .txt)
- **file_size**: File size in bytes
- **last_modified**: Last modification timestamp (ISO format)
- **content_hash**: SHA256 hash of document content
- **author**: Author name extracted from content or source URL path (string or null)
- **title**: Document title from content or cleaned filename (string)
- **publication_date**: Publication date in YYYY-MM-DD format (ISO string or null)
- **tags**: Array of relevant keywords/topics from content (array of strings)

## 🎮 Usage

### Basic Commands

- **General Chat**: Type normally to chat without knowledge base
- **Knowledge Base Search**: Use `@knowledgebase` to trigger RAG search
- **Hybrid Search**: Use natural language filters: `@knowledgebase papers by Smith from 2023`
- **System Info**: `/info` - Display current configuration including hybrid search status
- **Clear History**: `/clear` - Reset conversation
- **Exit**: `/bye` - Exit the application
- **View Documents**: `/doc <number>` - See full document content

### Example Session

```bash
$ python3 main.py

🤖 Welcome to RAG CLI Chatbot!

You: Hello, how are you?
Assistant: Hello! I'm doing well, thank you for asking...

You: @knowledgebase What is Python?
🔍 Searching knowledge base with 'Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has extensive libraries...'...
💭 Thinking with LLM prompt 'Explain what Python is based on the provided context, including...'...
Assistant: Based on the knowledge base, Python is a high-level programming language...

📚 Retrieved from knowledge base:
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Source                                      ┃ Score  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1 │ python_docs.md                              │ 0.892  │
│ 2 │ programming_guide.txt                       │ 0.845  │
└───┴─────────────────────────────────────────────┴────────┘
💡 Use /doc <number> to see full content

You: Tell me more about its web frameworks
💭 Thinking with LLM prompt 'Provide more details about Python web frameworks based on conte...'...
Assistant: Based on our previous discussion about Python, here are the popular web frameworks...

You: What about Django specifically?
💭 Thinking with LLM prompt 'Explain Django specifically based on context in previous conver...'...
Assistant: Django, as mentioned in our conversation, is a high-level Python web framework...

You: @knowledgebase papers by John Smith about machine learning from 2023
🔍 Searching knowledge base with 'machine learning' (filters: author=John Smith, tags=machine learning, publication_date=2023)...
💭 Thinking with LLM prompt 'Based on the provided context, provide information about machine learning from John Smith's papers...'...
Assistant: Based on John Smith's 2023 papers about machine learning from the knowledge base...

📚 Retrieved from knowledge base:
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Source                                      ┃ Score  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1 │ ML_Fundamentals_Smith_2023.pdf             │ 0.934  │
│ 2 │ Deep_Learning_Smith_2023.pdf               │ 0.889  │
└───┴─────────────────────────────────────────────┴────────┘
💡 Use /doc <number> to see full content
```

**Key Features Demonstrated:**
- **Knowledge Base Search**: `@knowledgebase` triggers RAG search with optimized queries
- **Hybrid Search**: Natural language metadata filtering combines semantic search with structured filters
- **HyDE Retrieval**: Shows hypothetical document generation for semantic search (example uses HyDE strategy)
- **Conversational Follow-ups**: "Tell me more..." uses conversation history without re-searching
- **Smart Context Switching**: System intelligently chooses between knowledge base and conversation context

## 🧪 Testing

### Unit Tests

To run unit tests only (mocked dependencies, fast):

```bash
# Activate virtual environment
source venv/bin/activate

# Run unit tests only
./doit.sh test
# OR directly: pytest tests/
```

### Integration Tests

To run integration tests with real LLM providers (requires valid config):

```bash
# Ensure config/config.yaml is properly configured with real API keys
source venv/bin/activate

# Run integration tests
./doit.sh integration-test
# OR directly: pytest integration_tests/ -m integration
```

**Integration Test Requirements:**
- Valid `config/config.yaml` with working LLM provider settings
- Internet connection for cloud LLM providers (Gemini)
- Proper API keys configured

### Full Test Suite

To run all tests (unit + integration):

```bash
source venv/bin/activate
pytest
```

**Test Coverage:**
- **68 unit tests** - Core logic with mocked dependencies  
- **8 integration tests** - Real LLM provider functionality
- **Comprehensive coverage** - QueryRewriter, CLI, fallback behavior, edge cases

## 🛠️ Development Commands

For common development tasks, use the `doit.sh` script:

```bash
./doit.sh <command>
```

Available commands:

- `test`: Runs the `pytest` test suite.
- `integration-test`: Runs integration tests with real LLM providers (requires valid config.yaml).
- `lint`: Runs the `ruff` linter without applying fixes.
- `lint-fix`: Runs the `ruff` linter and attempts to auto-fix issues.

## 📁 Project Structure

```
rag-cli-chatbot/
├── src/                    # Source code
│   ├── cli.py             # Main CLI interface
│   ├── config_manager.py  # Configuration management
│   ├── embedding_client.py # Embedding generation
│   ├── llm_client.py      # LLM interaction
│   ├── query_rewriter.py  # LLM-based query transformation
│   ├── qdrant_db.py       # Vector database client
│   └── logging_config.py  # Logging setup
├── config/                # Configuration files
│   ├── config.sample.yaml # Sample configuration template
│   └── config.yaml        # Main configuration (user-created)
├── tests/                 # Unit tests
│   ├── test_config_manager.py
│   ├── test_phase1_clients.py
│   ├── test_query_rewriter.py
│   └── test_cli_mocked.py
├── integration_tests/     # Integration tests with real LLM providers
│   ├── test_query_rewriter_integration.py
│   └── test_cli_integration.py
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── doit.sh              # Development commands
└── README.md            # This file
```

## 🔧 Development

### Phase Implementation Status

- ✅ **Phase 0**: Project setup, configuration, logging
- ✅ **Phase 1**: Core service integration (LLM, Embeddings, Qdrant)
- ✅ **Phase 2**: MVP CLI with RAG and fallback logic
- ✅ **Phase 3**: LLM-based query transformation and routing
- ✅ **Phase 4**: HyDE (Hypothetical Document Embeddings) dual retrieval strategies

### Key Features Implemented

- [x] Configuration management with environment variable support
- [x] Multiple LLM providers (Ollama, Gemini)
- [x] Multiple embedding providers (SentenceTransformers, Ollama, Gemini) 
- [x] Qdrant vector database integration
- [x] RAG trigger detection (@knowledgebase)
- [x] **Dual retrieval strategies** - Rewrite (keyword-based) and HyDE (semantic)
- [x] **LLM-based query transformation** for improved search accuracy
- [x] **Conversational context awareness** - intelligent follow-up handling
- [x] **Configurable system prompts** - customize assistant role and personality
- [x] Structured prompt generation with context instructions
- [x] Smart fallback system with edge case handling
- [x] Rich CLI interface with formatted output and progress indicators
- [x] Conversation history management
- [x] Document detail viewing
- [x] Comprehensive test suite with integration tests (76 total tests)

## 🔮 Future Enhancements

- **Document Ingestion Pipeline**: Automated document processing and chunking
- **Hybrid Search**: Combine vector and keyword search with reciprocal rank fusion
- **Re-ranking**: Advanced result scoring and ordering with cross-encoder models
- **Implicit RAG**: Automatic knowledge base routing without trigger phrases
- **Query Expansion**: Multi-query generation for comprehensive retrieval
- **Web Interface**: Browser-based chat interface

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**: Ensure virtual environment is activated
2. **Qdrant connection failed**: Check if Qdrant server is running
3. **LLM timeout**: Adjust timeout settings in config
4. **Empty knowledge base**: Ensure documents are properly ingested

### Debug Mode

Enable debug logging in `config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration examples
