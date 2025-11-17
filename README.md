# Plagiarism Detection System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

A comprehensive plagiarism detection system powered by advanced semantic similarity analysis and AI-driven deep inspection capabilities. The system provides both web interface and command-line tools for detecting potential plagiarism in academic texts.

## 🚀 Features

### Core Detection Capabilities
- **Semantic Similarity Analysis**: Uses state-of-the-art sentence transformers for accurate text similarity detection
- **Multi-level Detection**: Supports both sentence-level and paragraph-level plagiarism detection
- **Citation Recognition**: Intelligent detection and penalty adjustment for properly cited content
- **Cross-lingual Support**: Detects plagiarism across different languages
- **High-Performance Processing**: GPU acceleration and CPU multi-threading support

### AI-Powered Analysis
- **Smart Agent Integration**: Uses Large Language Models (LLM) for deep analysis of high-risk text pairs
- **Dual-Phase Analysis**: Provides both prosecution and defense perspectives for comprehensive evaluation
- **Evidence Extraction**: Automatically identifies and highlights key evidence of potential plagiarism
- **Contextual Reasoning**: Generates detailed explanations and reasoning for each detection

### User-Friendly Interface
- **Web Interface**: Intuitive Streamlit-based web application
- **Multiple Detection Modes**: 
  - Target file detection (compare one file against multiple references)
  - All-files comparison (detect mutual similarities among multiple files)
- **Interactive Visualization**: Color-coded highlighting with similarity scores
- **Comprehensive Reporting**: Export results in CSV, JSON, and Word formats

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Libraries

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

The system requires the following packages:

- **sentence-transformers**: For semantic text embeddings and similarity computation
- **faiss-cpu**: High-performance similarity search and clustering library
- **numpy**: Numerical computing support
- **streamlit**: Web interface framework
- **python-docx**: Word document generation for reports
- **openai**: LLM integration for AI agent analysis

### Optional Dependencies

For GPU acceleration (recommended for large datasets):
```bash
# Replace faiss-cpu with faiss-gpu if you have CUDA support
pip uninstall faiss-cpu
pip install faiss-gpu
```

## 🖥️ Usage

### Web Interface (Recommended)

1. **Start the Web Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Interface**: 
   Open your browser and navigate to `http://localhost:8501`

3. **Upload Files**:
   - Choose detection mode (Target file or All files comparison)
   - Upload text files (.txt or .md format)
   - Configure detection parameters in the sidebar

4. **Run Detection**:
   - Click "Start Detection" to begin analysis
   - View results in the "Comparison Analysis" tab
   - Export reports in various formats

### Command Line Interface

For batch processing or automated workflows:

```bash
python plagiarism_checker/cli.py --input-dir /path/to/submissions --output-dir /path/to/results
```

### Python API

For integration into other applications:

```python
from plagiarism_checker.pipeline import PlagiarismPipeline, PipelineConfig
from pathlib import Path

# Configure the pipeline
config = PipelineConfig(
    submissions_dir=Path("./submissions"),
    output_dir=Path("./results"),
    similarity_threshold=0.82,
    enable_citation_check=True,
    enable_agent=True
)

# Run detection
pipeline = PlagiarismPipeline(config)
sent_stats, sent_details = pipeline.run()
```

## ⚙️ Configuration

### Basic Configuration

Key parameters that can be adjusted:

- **Similarity Threshold**: Minimum similarity score to flag potential plagiarism (default: 0.82)
- **Paragraph Threshold**: Threshold for paragraph-level detection (default: 0.75)
- **Device Selection**: Choose between CPU, GPU, or auto-detection
- **Parallel Processing**: Enable multi-threading for faster processing

### AI Agent Configuration

To enable AI-powered analysis, create an `api_config.json` file:

```json
{
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key": "your-api-key-here",
    "model": "gpt-3.5-turbo"
}
```

Supported providers:
- OpenAI API
- Azure OpenAI
- Other OpenAI-compatible APIs

### Advanced Settings

- **Citation Recognition**: Automatically detects and reduces penalties for properly cited content
- **Cross-lingual Detection**: Uses multilingual models for detecting plagiarism across languages
- **Agent Threshold**: Minimum risk score to trigger AI analysis (default: 0.70)
- **Dual-phase Analysis**: Enables both prosecution and defense analysis perspectives

## 📊 Output Formats

The system generates multiple types of reports:

### 1. Summary Reports
- **CSV Format**: Tabular summary of all detected pairs with key metrics
- **JSON Format**: Detailed results with complete metadata
- **Word Summary**: Executive summary with key findings

### 2. Detailed Reports
- **Word Detailed Report**: Comprehensive analysis with highlighted text passages
- **Interactive Web View**: Real-time visualization with hover tooltips
- **Agent Reports**: AI-generated analysis and recommendations

### 3. Metrics Included
- Similarity scores (raw and adjusted for citations)
- Text coverage percentages
- Source specificity measures
- Risk level classifications
- Citation label distributions

## 🎯 Use Cases

### Academic Institutions
- **Student Assignment Checking**: Compare student submissions against reference materials
- **Batch Processing**: Check multiple assignments simultaneously for mutual plagiarism
- **Research Paper Analysis**: Verify originality of research submissions

### Content Management
- **Web Content Verification**: Check for duplicate or plagiarized online content
- **Document Comparison**: Compare versions of documents for changes and similarities
- **Quality Assurance**: Ensure content originality in publishing workflows

### Legal and Professional Services
- **Document Analysis**: Compare legal documents for similarity
- **Report Verification**: Check professional reports for potential plagiarism
- **Evidence Analysis**: Analyze text evidence for similarity patterns

## 🔧 Technical Architecture

### Core Components

1. **Corpus Module**: Text preprocessing and document loading
2. **Embedder Module**: Sentence transformer integration and vector generation
3. **Similarity Module**: FAISS-based similarity search and pair detection
4. **Citation Module**: Reference detection and penalty calculation
5. **Agent Module**: LLM-powered analysis and reasoning
6. **Pipeline Module**: Orchestrates the entire detection workflow
7. **Reporting Module**: Multi-format output generation

### Performance Optimizations

- **Vector Indexing**: Uses FAISS for efficient similarity search
- **Parallel Processing**: Multi-threaded embedding generation
- **GPU Acceleration**: CUDA support for large-scale processing
- **Caching**: Intelligent caching of embeddings and results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest plagiarism_checker/test/`
4. Make your changes and submit a PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [documentation](#) for common solutions
2. Search [existing issues](../../issues) for similar problems
3. Create a [new issue](../../issues/new) with detailed information

## 🙏 Acknowledgments

- **Sentence Transformers** team for providing excellent pre-trained models
- **FAISS** team for high-performance similarity search capabilities
- **Streamlit** team for the intuitive web framework
- **OpenAI** for providing powerful language models for analysis

---

**Note**: This system is designed to assist in plagiarism detection and should be used as a tool to support human judgment rather than replace it. Always review flagged content manually to make final determinations about potential plagiarism.