# ICD-10 Extraction System Architecture

## Overview

The ICD-10 Extraction System is a multi-agent architecture that uses different AI approaches to extract medical codes from clinical text. The system is built using LangGraph for orchestration and LangChain for LLM interactions.

## Architecture Components

### 1. Core System (`src/icd10_extractor/`)

#### `extractor.py` - Main System
- **ICD10ExtractionSystem**: Main orchestrator class
- Manages the multi-agent workflow using LangGraph
- Coordinates between different extraction agents
- Handles result aggregation and CSV export

#### `agents.py` - Agent Definitions
- **AgentState**: TypedDict defining the shared state structure
- Contains all data passed between agents during execution
- Tracks extraction results, performance metrics, and debug information

#### `retrievers.py` - Retrieval Systems
- **RetrieverManager**: Manages FAISS and BM25 retrievers
- Sets up vector database for semantic search
- Configures BM25 for keyword-based retrieval
- Handles database persistence and loading

#### `utils.py` - Utility Functions
- Helper functions for file I/O and directory management
- Patient note loading with fallback to default content
- Directory creation utilities

### 2. Agent Architecture

The system employs 5 specialized agents:

1. **GPT-4 Extractor** (Simulated)
   - Uses GPT-4o-mini as a proxy for GPT-4
   - Direct extraction from clinical text
   - Temperature: 0.1 for consistency

2. **GPT-4o-mini Extractor**
   - Direct GPT-4o-mini extraction
   - Provides baseline performance comparison
   - Same prompt template as GPT-4

3. **GPT-4o-mini Instance 2**
   - Second instance for result validation
   - Tests consistency across multiple runs
   - Independent extraction path

4. **RAG Extractor**
   - Uses FAISS vector database for context retrieval
   - Semantic similarity search (k=10)
   - Context-aware extraction with confidence scores
   - Leverages ICD-10 code embeddings

5. **BM25 Extractor**
   - Keyword-based retrieval using BM25 algorithm
   - Complementary to semantic search
   - Good for exact term matching
   - Fast retrieval for large datasets

### 3. Data Flow

```
Clinical Text Input
        ↓
   Start Node
        ↓
   GPT-4 Agent ──────────┐
        ↓                │
   GPT-4o-mini Agent     │
        ↓                │
   GPT-4o-mini (2nd)     │ → Parallel Processing
        ↓                │   (Future Enhancement)
   RAG Agent ────────────┤
        ↓                │
   BM25 Agent ───────────┘
        ↓
   Result Aggregation
        ↓
   CSV Export & Display
```

### 4. State Management

The `AgentState` maintains:
- **Input**: `patient_note` (clinical text)
- **Outputs**: Lists of extracted codes per agent
- **Metadata**: Token usage, retry counts, confidence scores
- **Debug**: Error information and execution logs

### 5. Configuration System

- **YAML Configuration**: `config/config.yaml`
- Model parameters (temperature, max tokens)
- Data paths and output settings
- Agent enable/disable flags
- Performance tuning parameters

### 6. Retrieval Systems

#### FAISS Vector Database
- Stores ICD-10 codes as vector embeddings
- Uses OpenAI embeddings for semantic similarity
- Persistent storage with automatic loading
- Similarity search with configurable k

#### BM25 Retriever
- Term frequency-based ranking
- Good for exact medical terminology
- Complementary to semantic search
- Fast keyword matching

### 7. Output Processing

- **Structured Results**: JSON-like dictionaries with codes and descriptions
- **CSV Export**: Flattened format for analysis
- **Performance Metrics**: Token usage and timing data
- **Confidence Scores**: Available for RAG extractions

## Design Patterns

### 1. Multi-Agent Pattern
- Independent agents with specialized capabilities
- Shared state for coordination
- Sequential execution with potential for parallelization

### 2. Retrieval-Augmented Generation (RAG)
- External knowledge base integration
- Context injection for improved accuracy
- Semantic and keyword-based retrieval

### 3. Configuration-Driven Design
- YAML-based configuration
- Environment-specific settings
- Easy parameter tuning

### 4. Modular Architecture
- Separation of concerns
- Pluggable components
- Easy testing and maintenance

## Performance Considerations

### 1. Token Optimization
- Efficient prompt templates
- Context window management
- Token usage tracking

### 2. Caching Strategy
- FAISS database persistence
- Embedding reuse
- Result caching (future enhancement)

### 3. Error Handling
- Graceful degradation
- Retry mechanisms
- Comprehensive logging

## Future Enhancements

### 1. Parallel Processing
- Concurrent agent execution
- Reduced total processing time
- Resource optimization

### 2. Advanced Retrieval
- Hybrid search combining FAISS and BM25
- Re-ranking mechanisms
- Query expansion techniques

### 3. Model Diversity
- Support for different LLM providers
- Model ensemble approaches
- Specialized medical models

### 4. Evaluation Framework
- Ground truth comparison
- Inter-agent agreement metrics
- Performance benchmarking

## Dependencies

- **LangChain**: LLM orchestration and prompting
- **LangGraph**: Multi-agent workflow management
- **FAISS**: Vector similarity search
- **OpenAI**: Language model API
- **Pandas**: Data manipulation and CSV handling
- **PyYAML**: Configuration file parsing
