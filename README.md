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
  retrieval_strategy: "hyde"  # Generate 3 multi-persona hypothetical documents for semantic search
  trigger_phrase: "@knowledgebase"
  top_k: 10
  min_score: 0.3
```

**Strategy Comparison:**

| Strategy | Speed | Best For | Search Method |
|----------|-------|----------|---------------|
| **Rewrite** | Fast | Factual queries, keyword matching | `"neural networks training"` |
| **HyDE** | Slower | Conceptual questions, semantic understanding | 3 perspectives: Professor, Teacher, Student views |

**When to Use:**
- **Rewrite**: Start here for general use, good performance
- **HyDE**: Switch if you need better retrieval of conceptually related content

### Multi-Persona HyDE Generation

The HyDE strategy now generates **3 different hypothetical documents** from different expert perspectives for better semantic coverage:

**Science Topics** (AI, machine learning, physics, etc.):
- **Professor perspective**: Technical, research-focused with advanced terminology
- **Teacher perspective**: Educational, accessible explanations with examples  
- **Student perspective**: Learning-focused, discovery-oriented with questions

**Business Topics** (management, strategy, finance, etc.):
- **Director perspective**: Strategic, high-level decision-making focus
- **Manager perspective**: Operational, practical implementation approach
- **Assistant perspective**: Detailed, process-oriented support tasks

**Other Topics**:
- **Expert perspective**: Authoritative, comprehensive professional knowledge
- **Educator perspective**: Teaching-focused, well-structured information
- **Learner perspective**: Curious, exploratory with growth mindset

**Example Multi-Persona Generation for "neural networks"**:
```
Professor: "Neural networks utilize backpropagation algorithms for weight optimization..."
Teacher: "Neural networks learn by adjusting connections between artificial neurons..."
Student: "I'm studying how neural networks mimic the human brain to recognize patterns..."
```

**Benefits:**
- **Enhanced Semantic Coverage**: Multiple viewpoints capture diverse aspects of topics
- **Improved Retrieval Accuracy**: Matches documents from different conceptual angles
- **Better Context Understanding**: Covers technical, educational, and practical perspectives
- **Robust Performance**: Works well across different document types and writing styles

This multi-persona approach is especially effective for complex queries where different documents may approach the same topic from varying expertise levels or professional contexts.

### Hybrid Search with Intent-Based Patterns

**Enhanced RAG System**: The application now uses intent-based pattern detection to intelligently handle different types of queries with natural language metadata filtering.

**Enable Hybrid Search**:
```yaml
rag:
  use_hybrid_search: true  # Enable natural language metadata filtering
  retrieval_strategy: "rewrite"  # Works with both rewrite and hyde
  trigger_phrase: "@knowledgebase"
```

**Query Patterns:**

The system automatically detects three distinct query patterns based on user intent:

#### 📋 Pattern 1: Pure Search (Document Discovery)
**Intent**: Find and browse documents  
**Behavior**: Extracts metadata filters + returns document summaries with question suggestions

**Examples:**
- `search @knowledgebase on machine learning`
- `@knowledgebase find papers by Smith from 2024`
- `get @knowledgebase articles about Python`
- `locate @knowledgebase documents tagged AI, not by Johnson`

#### 🔍 Pattern 2: Search + Action (Find Then Analyze)  
**Intent**: Find documents and perform analysis on them  
**Behavior**: Extracts metadata filters + performs specified action on retrieved results

**Examples:**
- `search @knowledgebase on AI, explain the key concepts`
- `@knowledgebase find papers by Smith and summarize the findings`
- `get @knowledgebase docs on Python, what are the benefits`
- `@knowledgebase retrieve neural network research, compare the approaches`

#### 💬 Pattern 3: Direct Questions (Knowledge Consultation)
**Intent**: Ask direct questions about topics  
**Behavior**: No metadata extraction - treats all context as semantic information

**Examples:**
- `@knowledgebase what is machine learning`
- `@knowledgebase explain the benefits of Python programming`
- `@knowledgebase compare REST vs GraphQL APIs`
- `@knowledgebase how does neural network training work`

**Key Innovation: Intent-Based Detection**
- **No rigid syntax**: Uses natural language understanding instead of word-order rules
- **Flexible expressions**: Supports various connectors ("and", "then", commas, contextual flow)
- **Eliminates ambiguity**: `@knowledgebase search on AI` and `search @knowledgebase on AI` now behave consistently

**Three-Filter System:**

| Filter Type | Intent Detection | Behavior | Example |
|------------|------------------|----------|---------|
| **Hard Filters** | Restrictive intent: "only", "exclusively", "just", "must be", "limited to", "solely" | Must match (excludes non-matching) | `papers only from 2024` |
| **Negation Filters** | Exclusion intent: "not", "without", "except", "excluding", "avoid", "skip" | Must NOT match (excludes matching) | `research not by Johnson` |  
| **Soft Filters** | All other metadata mentions | Boost if match (doesn't exclude) | `papers by Smith about AI` |

**Key Innovation**: Uses **intent-based detection** rather than exact keyword matching, supporting natural language variations like `"only authored by Smith"` being correctly detected as a hard filter.

**Natural Language Metadata Examples:**

| User Query | Extracted Filters | Pattern | Result |
|------------|------------------|---------|---------|
| `search @knowledgebase papers by Smith` | `{"author": "Smith"}` (soft) | Pattern 1 | Document summaries |
| `@knowledgebase find articles ONLY from 2024` | `{"publication_date": "2024"}` (hard) | Pattern 1 | Only 2024 docs |
| `get @knowledgebase ML papers, not by Johnson` | `{"tags": ["ML"]}` (soft), `{"author": "Johnson"}` (negation) | Pattern 1 | ML papers excluding Johnson |
| `search @knowledgebase on Python and explain benefits` | `{"tags": ["python"]}` (soft) | Pattern 2 | Find + explain |
| `@knowledgebase what is machine learning by Smith` | No filters extracted | Pattern 3 | Direct question |

**Requirements:**
- Qdrant collection must have payload indexes for: `tags`, `author`, `publication_date`  
- These indexes are created during document ingestion
- System logs warnings if indexes are missing (performance impact)

**Benefits:**
- **Intent-Aware**: Automatically chooses the right behavior based on user intent
- **Natural Interface**: No complex syntax or trigger words required
- **Flexible Language**: Supports various phrasings and connectors
- **Precision**: Combines semantic search with intelligent metadata filtering
- **Performance**: Uses Qdrant indexes for efficient filtering

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
- **Knowledge Base Access**: Use `@knowledgebase` to trigger RAG search with intent-based patterns
- **System Info**: `/info` - Display current configuration including hybrid search status  
- **Help**: `/help` - Show comprehensive query pattern examples
- **Clear History**: `/clear` - Reset conversation
- **Exit**: `/bye` - Exit the application
- **View Documents**: `/doc <number>` - See full document content

### Query Patterns Usage

**📋 Document Discovery** (Pattern 1):
- `search @knowledgebase on neural networks`
- `@knowledgebase find papers by Dr. Smith from 2024`
- `get @knowledgebase articles about Python, tagged machine learning`

**🔍 Search + Analysis** (Pattern 2):  
- `search @knowledgebase on AI research, explain the main trends`
- `@knowledgebase find Smith's papers and summarize the key findings`
- `get @knowledgebase Python docs, what are the best practices`

**💬 Direct Questions** (Pattern 3):
- `@knowledgebase what is the difference between supervised and unsupervised learning`
- `@knowledgebase explain how neural networks work`
- `@knowledgebase compare the benefits of Python vs Java for data science`

### Example Session

```bash
$ python3 main.py

🤖 Welcome to RAG CLI Chatbot!

You: Hello, how are you?
Assistant: Hello! I'm doing well, thank you for asking...

You: @knowledgebase what is Python?
🔍 Searching knowledge base with 'Python programming language'...
💭 Thinking with LLM prompt 'Based on the provided context, explain what Python is...'...
Assistant: Based on the knowledge base, Python is a high-level programming language known for its simplicity and readability...

📚 Retrieved from knowledge base:
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Source                                      ┃ Score  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1 │ python_docs.md                              │ 0.892  │
│ 2 │ programming_guide.txt                       │ 0.845  │
└───┴─────────────────────────────────────────────┴────────┘
💡 Use /doc <number> to see full content

You: search @knowledgebase on neural networks
🔍 Searching knowledge base with 'neural networks'...
💭 Thinking with SEARCH_SUMMARY_MODE...
Assistant: 📋 **Document Summary**: Found 5 documents about neural networks covering deep learning fundamentals, training algorithms, and practical applications...

**Key Topics**: Backpropagation, CNN architectures, Training optimization, Real-world applications

**Question Suggestions**:
- What are the main types of neural network architectures?
- How does backpropagation work in neural network training?
- What are common applications of convolutional neural networks?
- What optimization techniques improve neural network performance?

You: @knowledgebase find papers by Smith and explain the key findings
🔍 Searching knowledge base with 'papers' (filters: author=Smith)...
💭 Thinking with LLM prompt 'Based on the provided context, explain the key findings...'...
Assistant: Based on Smith's papers from the knowledge base, here are the key findings across their research...

📚 Retrieved from knowledge base:
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Source                                      ┃ Score  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1 │ Smith_ML_Research_2024.pdf                  │ 0.934  │
│ 2 │ Smith_Deep_Learning_2024.pdf               │ 0.889  │
└───┴─────────────────────────────────────────────┴────────┘
💡 Use /doc <number> to see full content
```

**Key Features Demonstrated:**
- **Intent-Based Patterns**: System automatically detects search vs. question vs. search+action intents
- **Pattern 3 (Direct Question)**: `@knowledgebase what is Python?` - treats metadata as semantic context
- **Pattern 1 (Pure Search)**: `search @knowledgebase on neural networks` - returns document summaries + suggestions  
- **Pattern 2 (Search+Action)**: `@knowledgebase find papers by Smith and explain` - searches then analyzes
- **Natural Language Filtering**: Automatically extracts author, date, and tag filters from natural language
- **Three-Filter System**: Hard/negation/soft filters with intelligent classification
- **Flexible Language**: No rigid syntax - supports various phrasings and connectors

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

**Full Integration Test Suite** (slow, >4 mins):
```bash
# Ensure config/config.yaml is properly configured with real API keys
source venv/bin/activate

# Run all integration tests
./doit.sh integration-test
# OR directly: pytest integration_tests/
```

**Specific Integration Tests** (faster, targeted testing):
```bash
source venv/bin/activate

# Intent-based filtering & query cleaning
pytest integration_tests/test_query_rewriter_integration.py -v

# HyDE multi-persona functionality  
pytest integration_tests/test_query_rewriter_multi_persona_hyde.py -v

# Vector database operations
pytest integration_tests/test_qdrant_integration.py -v

# CLI workflow testing
pytest integration_tests/test_cli_integration.py -v
```

**Integration Test Requirements:**
- Valid `config/config.yaml` with working LLM provider settings
- Internet connection for cloud LLM providers (Gemini)
- Proper API keys configured
- **Uses ConfigManager**: Properly handles environment variables like `GEMINI_API_KEY`

### Query Pattern Testing

Test how the query rewriter transforms different patterns:

```bash
source venv/bin/activate

# Using doit.sh command (recommended)
./doit.sh check-query-rewriter '@knowledgebase what is machine learning'
./doit.sh check-query-rewriter 'search @knowledgebase on AI research'
./doit.sh check-query-rewriter '@knowledgebase find papers by Smith and summarize'

# Direct usage with different strategies
RAG_STRATEGY=hyde python check_query_rewriter.py '@knowledgebase explain neural networks'
RAG_STRATEGY=rewrite python check_query_rewriter.py 'search @knowledgebase on Python'
```

The tool shows the complete transformation including:
- Detected query pattern (1, 2, or 3)
- Extracted metadata filters (hard/negation/soft)
- Generated embedding text
- Final LLM prompt
- Strategy used (Rewrite vs HyDE)

### Full Test Suite

To run all tests (unit + integration):

```bash
source venv/bin/activate
pytest
```

**Test Coverage:**
- **149 unit tests** - Core logic with mocked dependencies  
- **59 integration tests** - Real LLM provider functionality including multi-persona HyDE
- **Comprehensive coverage** - QueryRewriter, CLI, fallback behavior, multi-persona generation, edge cases

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
- `check-query-rewriter 'query'`: Test query rewriter with a specific query to see pattern detection and filter extraction.

**Examples:**
```bash
./doit.sh test
./doit.sh lint-fix
./doit.sh check-query-rewriter 'search @knowledgebase on AI, only by Smith'
```

## 📁 Project Structure

```
rag-cli-chatbot/
├── src/                    # Source code
│   ├── cli.py             # Main CLI interface with intent-based patterns
│   ├── config_manager.py  # Configuration management
│   ├── embedding_client.py # Embedding generation
│   ├── llm_client.py      # LLM interaction
│   ├── query_rewriter.py  # Intent-based query transformation & pattern detection
│   ├── search_service.py  # Unified search with soft filtering
│   ├── qdrant_db.py       # Vector database client
│   └── logging_config.py  # Logging setup
├── config/                # Configuration files
│   ├── config.sample.yaml # Sample configuration template
│   └── config.yaml        # Main configuration (user-created)
├── tests/                 # Unit tests
│   ├── test_config_manager.py
│   ├── test_phase1_clients.py
│   ├── test_query_rewriter.py
│   ├── test_search_service.py
│   └── test_cli_mocked.py
├── integration_tests/     # Integration tests with real LLM providers
│   ├── test_query_rewriter_integration.py
│   ├── test_query_rewriter_multi_persona_hyde.py
│   ├── test_soft_filtering_integration.py
│   └── test_cli_integration.py
├── main.py               # Application entry point
├── check_query_rewriter.py # Query pattern testing tool
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
- [x] **Intent-Based Query Patterns** - automatic detection of search vs. question vs. search+action intents
- [x] **Three-Pattern System** - Pure Search, Search+Action, Direct Questions with different behaviors
- [x] **Natural Language Metadata Filtering** - extract author, date, tag filters from conversational language
- [x] **Three-Filter Classification** - Hard/Negation/Soft filters with intelligent keyword detection
- [x] **Dual retrieval strategies** - Rewrite (keyword-based) and HyDE (semantic)
- [x] **Unified Search Service** - combines vector search with metadata filtering and soft boosting
- [x] **LLM-based query transformation** for improved search accuracy and pattern detection
- [x] **Conversational context awareness** - intelligent follow-up handling
- [x] **Configurable system prompts** - customize assistant role and personality
- [x] **Flexible Language Support** - no rigid syntax, supports various connectors and phrasings
- [x] Smart fallback system with edge case handling
- [x] Rich CLI interface with formatted output and comprehensive help examples
- [x] Query pattern testing tool (`check_query_rewriter.py`)
- [x] Conversation history management
- [x] Document detail viewing
- [x] Comprehensive test suite with integration tests (76+ total tests)

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
