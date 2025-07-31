# RAG CLI Chatbot

A command-line chatbot with Retrieval-Augmented Generation (RAG) capabilities. Supports knowledge base search using Qdrant vector database and various LLM providers.

## 🎯 Features

- **Interactive CLI Interface**: Rich terminal interface with colors and formatting
- **RAG Capabilities**: Search and retrieve information from knowledge base
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
🔍 Searching knowledge base...
Assistant: Based on the knowledge base, Python is a high-level programming language...

📚 Retrieved from knowledge base:
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Source                                      ┃ Score  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1 │ python_docs.md                              │ 0.892  │
│ 2 │ programming_guide.txt                       │ 0.845  │
└───┴─────────────────────────────────────────────┴────────┘
💡 Use /doc <number> to see full content
```

## 🧪 Testing

To run the full test suite, use `pytest`:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest
```

This will discover and run all tests in the `tests/` directory. For more detailed output, you can use `pytest -v`.

## 📁 Project Structure

```
rag-cli-chatbot/
├── src/                    # Source code
│   ├── cli.py             # Main CLI interface
│   ├── config_manager.py  # Configuration management
│   ├── embedding_client.py # Embedding generation
│   ├── llm_client.py      # LLM interaction
│   ├── qdrant_db.py       # Vector database client
│   └── logging_config.py  # Logging setup
├── config/                # Configuration files
│   ├── config.sample.yaml # Sample configuration template
│   └── config.yaml        # Main configuration (user-created)
├── tests/                 # Test files
│   ├── test_config_manager.py
│   ├── test_phase1_clients.py
│   └── test_cli_moked.py
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Development

### Phase Implementation Status

- ✅ **Phase 0**: Project setup, configuration, logging
- ✅ **Phase 1**: Core service integration (LLM, Embeddings, Qdrant)
- ✅ **Phase 2**: MVP CLI with RAG and fallback logic

### Key Features Implemented

- [x] Configuration management with environment variable support
- [x] Multiple LLM providers (Ollama, Gemini)
- [x] Multiple embedding providers (SentenceTransformers, Ollama, Gemini) 
- [x] Qdrant vector database integration
- [x] RAG trigger detection (@knowledgebase)
- [x] Confidence-based fallback system
- [x] Rich CLI interface with formatted output
- [x] Conversation history management
- [x] Document detail viewing
- [x] Comprehensive test suite

## 🔮 Future Enhancements

- **Document Ingestion Pipeline**: Automated document processing and chunking
- **Query Rewriting**: LLM-based query enhancement
- **Hybrid Search**: Combine vector and keyword search
- **Re-ranking**: Advanced result scoring and ordering
- **Implicit RAG**: Automatic knowledge base routing
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
