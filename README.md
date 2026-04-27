# DeepTutor

> An intelligent tutoring system powered by large language models, forked from [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor).

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

DeepTutor is an AI-powered tutoring assistant that helps users understand complex documents, papers, and educational materials through interactive Q&A, summarization, and guided learning.

## Features

- 📄 **Document Understanding** — Upload PDFs and get instant comprehension assistance
- 🤖 **Multi-LLM Support** — Compatible with OpenAI, Anthropic, and local models
- 💬 **Interactive Q&A** — Ask questions about your documents in natural language
- 🧠 **Knowledge Graph** — Builds a semantic graph of document concepts
- 🌐 **Multilingual** — Supports both English and Chinese interfaces
- 🐳 **Docker Ready** — Easy deployment with Docker Compose

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/DeepTutor.git
   cd DeepTutor
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

### Docker Deployment

```bash
docker compose up --build
```

The application will be available at `http://localhost:7860`.

## Configuration

Copy `.env.example` to `.env` and configure the following:

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | Yes (if using OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes (if using Claude) |
| `LLM_PROVIDER` | LLM backend (`openai`, `anthropic`, `ollama`) | Yes |
| `EMBEDDING_MODEL` | Embedding model name | Yes |
| `MAX_UPLOAD_SIZE_MB` | Maximum file upload size in MB | No (default: 50) |

For Chinese deployment, see `.env.example_CN`.

## Project Structure

```
DeepTutor/
├── app.py                  # Main application entry point
├── core/                   # Core logic modules
│   ├── document_processor.py
│   ├── knowledge_graph.py
│   └── llm_interface.py
├── ui/                     # Frontend components
├── utils/                  # Utility functions
├── tests/                  # Test suite
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Contributing

We welcome contributions! Please check our [issue templates](.github/ISSUE_TEMPLATE/) before opening an issue.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/)
4. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Original project: [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)
- Built with [LightRAG](https://github.com/HKUDS/LightRAG), [Gradio](https://gradio.app/), and [LangChain](https://langchain.com/)
