# 🔍 RepoMind — AI-Powered Code Repository Q&A Agent

RepoMind is a RAG (Retrieval-Augmented Generation) agent that lets you ask natural-language questions about any codebase. Point it at a repo, and it indexes every file, builds a vector store, and answers questions like *"Where is the authentication logic?"* or *"What does this function do?"* — with exact file references.

Built with **LangChain**, **FAISS**, **Sentence Transformers**, and **HuggingFace Inference API**. Runs entirely on free-tier tools — no OpenAI key needed.

---

## 🎯 Features

- **Natural Language Code Q&A** — Ask plain English questions about any codebase and get precise, file-referenced answers.
- **RAG Pipeline** — Retrieval-Augmented Generation ensures the LLM only answers based on your actual code, not hallucinated knowledge.
- **Language-Aware Chunking** — Code is split at function/class boundaries (not mid-line) using LangChain's language-specific splitters for Python, JavaScript, TypeScript, Java, Go, Rust, C++, and more.
- **FAISS Vector Search with MMR** — Uses Facebook's FAISS for fast similarity search, with Maximal Marginal Relevance to ensure diverse, non-redundant results.
- **Conversational Memory** — Supports follow-up questions with rolling chat history (last 10 exchanges).
- **Lightweight & Free** — Embeddings run locally on CPU (~80MB model). LLM inference uses HuggingFace's free API tier.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Your Repo  │────▶│  File Loader     │────▶│  Language-Aware  │
│  (any lang) │     │  (52 file types) │     │  Text Splitter   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Answer +   │◀────│  Qwen-7B LLM    │◀────│  FAISS Vector   │
│  Sources    │     │  (HuggingFace)   │     │  Store (MiniLM) │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            ▲
                            │
                    ┌───────┴───────┐
                    │  Your Question │
                    └───────────────┘
```

---

## ⚡ Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/repomind.git
cd repomind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your HuggingFace token (free)

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
# Linux / macOS
export HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"

# Windows (CMD)
set HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# Windows (PowerShell)
$env:HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

### 4. Run RepoMind

```bash
python agent.py /path/to/any/repo
```

---

## 💬 Example Usage

```
Step 1: Loading repository...
  Loaded 52 files (1 skipped for size)
Step 2: Splitting into chunks...
  Created 92 chunks from 52 files
Step 3: Building vector store...
  Vector store ready!
Step 4: Setting up LLM...
  Connected to Qwen/Qwen2.5-7B-Instruct

============================================================
  Code Agent Ready! Ask anything about the repo.
  Type 'quit' to exit, 'clear' to reset memory.
============================================================

You > Where is the authentication logic?

Agent > The authentication logic is in `src/auth/login.py`. The `authenticate_user()`
function on line 34 takes a username and password, hashes the password using bcrypt,
and compares it against the stored hash in the database...

  Sources: src/auth/login.py, src/middleware/auth_middleware.py

You > What does the validate_token() function do?

Agent > The `validate_token()` function in `src/auth/token.py` decodes a JWT token,
checks its expiration timestamp, and returns the user payload if valid...

  Sources: src/auth/token.py, src/utils/jwt_helper.py
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Framework** | LangChain | Orchestration, text splitting, vector store integration |
| **Vector Store** | FAISS (Facebook AI) | Fast nearest-neighbor similarity search |
| **Embeddings** | `all-MiniLM-L6-v2` | Converts code chunks into 384-dim vectors (runs locally) |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` | Answers questions via HuggingFace Inference API (free) |
| **Search Strategy** | MMR (Maximal Marginal Relevance) | Balances relevance with diversity in retrieved chunks |
| **Language** | Python 3.10+ | Core implementation |

---

## 📁 Supported File Types

Python, JavaScript, TypeScript, JSX, TSX, Java, Go, Rust, C, C++, C#, Ruby, PHP, Swift, Kotlin, Scala, Vue, Svelte, HTML, CSS, SCSS, SQL, Shell, YAML, TOML, JSON, Markdown, Dockerfile, Terraform, Protobuf, GraphQL, and more.

---

## ⚙️ Configuration

All configuration is at the top of `agent.py`:

```python
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"              # AI model for answers
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local embedding model
MAX_FILE_SIZE_KB = 200                                 # Skip files larger than this
CODE_EXTENSIONS = {".py", ".js", ".ts", ...}           # File types to index
```

### Alternative Free Models

| Model | Best For |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | General code analysis (default) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | Specialized code understanding |
| `codellama/CodeLlama-7b-Instruct-hf` | Code generation and explanation |
| `google/gemma-2-2b-it` | Faster responses, smaller model |

---

## 📂 Project Structure

```
repomind/
├── agent.py             # Main agent — all logic in one file
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🔮 Roadmap

- [ ] **Streamlit Web UI** — Browser-based interface instead of CLI
- [ ] **Ollama Support** — Fully offline mode with local LLMs
- [ ] **Persistent Index** — Save/load FAISS index to skip re-indexing
- [ ] **Git-Aware Queries** — "What changed in the last commit?"
- [ ] **Multi-Repo Support** — Index and query across multiple projects
- [ ] **Code Generation** — "Add error handling to this function"

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using LangChain, FAISS, and HuggingFace
</p>
