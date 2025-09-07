# ICD-10 Extraction System

A multi-agent system for extracting ICD-10 codes from clinical text using LangGraph, LangChain, and multiple retrieval methods including RAG with FAISS and BM25.

## Features

- **Multi-Agent Architecture**: Uses 5 different AI agents for comprehensive code extraction
- **Multiple Retrieval Methods**: RAG with FAISS vector database and BM25 retrieval
- **Configurable**: YAML-based configuration system
- **Command-Line Interface**: Easy-to-use CLI for batch processing
- **Performance Tracking**: Token usage and retry metrics
- **CSV Export**: Structured output for analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd icd10_singlefile
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Project Structure

```
icd10_singlefile/
├── src/
│   └── icd10_extractor/
│       ├── __init__.py
│       ├── agents.py          # Agent state definitions
│       ├── extractor.py       # Main extraction system
│       ├── retrievers.py      # FAISS and BM25 setup
│       └── utils.py           # Utility functions
├── data/
│   ├── icd10_2019.csv        # ICD-10 code database
│   ├── clinical_text_3.txt   # Sample clinical text
│   └── icd10_faiss_db/       # FAISS vector database
├── output/                   # Results and logs
├── config/
│   └── config.yaml          # Configuration file
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── setup.py                # Package setup
```

## Usage

### Basic Usage

Extract ICD-10 codes from the default clinical text:
```bash
python main.py
```

### Advanced Usage

Specify input and output files:
```bash
python main.py --input data/clinical_text_3.txt --output output/my_results.csv
```

Enable verbose logging:
```bash
python main.py --verbose
```

Use custom ICD-10 data:
```bash
python main.py --icd10-data path/to/your/icd10_data.csv
```

### Command-Line Options

- `--input, -i`: Path to clinical text file
- `--output, -o`: Output CSV file path (default: output/extraction_results.csv)
- `--icd10-data`: Path to ICD-10 CSV data file (default: data/icd10_2019.csv)
- `--verbose, -v`: Enable verbose logging

## Agents

The system uses 5 different agents:

1. **GPT-4 Extractor**: Primary extraction using GPT-4o-mini (simulating GPT-4)
2. **GPT-4o-mini Extractor**: Direct GPT-4o-mini extraction
3. **GPT-4o-mini (Instance 2)**: Second instance for comparison
4. **RAG Extractor**: Uses FAISS vector database for context-aware extraction
5. **BM25 Extractor**: Uses BM25 retrieval for keyword-based extraction

## Configuration

Edit `config/config.yaml` to customize:

- Model parameters (temperature, max tokens)
- Data paths
- Output settings
- Retriever configurations
- Agent settings

## Output

The system generates:

- **Console Output**: Summary of extracted codes and performance metrics
- **CSV File**: Detailed results with all extracted codes, descriptions, and metadata
- **Performance Metrics**: Token usage and retry counts per agent

## Example Output

```
============================================================
EXTRACTION RESULTS SUMMARY
============================================================

GPT-4 Results: 27 codes
  1. J44.9: Chronic obstructive pulmonary disease, unspecified
  2. E11.9: Type 2 diabetes mellitus without complications
  3. I10: Essential (primary) hypertension
  4. N18.32: Chronic kidney disease, stage 3b
  5. I50.22: Heart failure with reduced ejection fraction
  ... and 22 more codes

RAG Results: 17 codes
  1. J44.1: Chronic obstructive pulmonary disease with acute exacerbation
  2. J44.0: Chronic obstructive pulmonary disease with acute lower respiratory infection
  3. I50.9: Heart failure, unspecified
  ... and 14 more codes

============================================================
PERFORMANCE METRICS
============================================================
ICD-10 Extractor (GPT-4): 1160 tokens, 1 retries
ICD-10 Extractor (RAG): 1287 tokens, 1 retries
ICD-10 Extractor (BM25): 1104 tokens, 1 retries
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.
