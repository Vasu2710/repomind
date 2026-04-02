"""
AI Code Repository Agent
========================
An AI-powered coding assistant that lets you upload a repo
and ask questions about the codebase using LangChain + HuggingFace.

Requirements:
    pip install langchain-text-splitters langchain-community langchain-core
    pip install langchain-huggingface faiss-cpu sentence-transformers
    pip install huggingface_hub requests

Setup:
    set HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
    (Get a free token at https://huggingface.co/settings/tokens)
"""

import os
import sys
import requests
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ── Configuration ──────────────────────────────────────────────────────────

# Free HuggingFace model (tested and working with chat completions API)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Free embedding model (runs locally, ~80MB download on first run)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace chat completions endpoint
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".vue", ".svelte", ".html", ".css", ".scss",
    ".sql", ".sh", ".bash", ".yaml", ".yml", ".toml", ".json",
    ".md", ".txt", ".env.example", ".dockerfile", "Dockerfile",
    ".tf", ".proto", ".graphql",
}

# Maximum file size to index (skip huge generated files)
MAX_FILE_SIZE_KB = 200


# ── HuggingFace Chat LLM ──────────────────────────────────────────────────

class HuggingFaceChatLLM:
    """
    Simple wrapper around HuggingFace's chat completions API.
    Works with models that support the /v1/chat/completions endpoint.
    """

    def __init__(self, model: str, token: str, temperature: float = 0.2, max_tokens: int = 1024):
        self.model = model
        self.token = token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def chat(self, system_prompt: str, user_message: str, chat_history: list = None) -> str:
        """Send a chat completion request and return the response text."""
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add chat history
        if chat_history:
            messages.extend(chat_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(HF_API_URL, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"Unexpected API response: {data}")


# ── Repository Loader ─────────────────────────────────────────────────────

def load_repository(repo_path: str) -> List[Document]:
    """Walk through a repository and load all code files as Documents."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise ValueError(f"'{repo_path}' is not a valid directory.")

    documents: List[Document] = []
    skipped = 0

    for file_path in repo.rglob("*"):
        parts = file_path.parts
        if any(
            p.startswith(".") or p in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")
            for p in parts
        ):
            continue

        if not file_path.is_file():
            continue

        ext = file_path.suffix
        if ext not in CODE_EXTENSIONS and file_path.name not in CODE_EXTENSIONS:
            continue

        if file_path.stat().st_size > MAX_FILE_SIZE_KB * 1024:
            skipped += 1
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            relative = file_path.relative_to(repo)
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(relative),
                    "extension": ext,
                    "filename": file_path.name,
                    "full_path": str(file_path),
                },
            )
            documents.append(doc)
        except Exception as e:
            print(f"  Warning: Skipped {file_path.name}: {e}")

    print(f"  Loaded {len(documents)} files ({skipped} skipped for size)")
    return documents


# ── Text Splitting ─────────────────────────────────────────────────────────

LANG_MAP = {
    ".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS,
    ".jsx": Language.JS, ".tsx": Language.TS, ".java": Language.JAVA,
    ".go": Language.GO, ".rs": Language.RUST, ".cpp": Language.CPP,
    ".c": Language.CPP, ".rb": Language.RUBY, ".scala": Language.SCALA,
    ".swift": Language.SWIFT, ".md": Language.MARKDOWN, ".html": Language.HTML,
}


def split_documents(documents: List[Document]) -> List[Document]:
    """Split code files using language-aware splitters."""
    all_chunks: List[Document] = []

    for doc in documents:
        ext = doc.metadata.get("extension", "")
        lang = LANG_MAP.get(ext)

        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1500, chunk_overlap=200,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=200,
            )

        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"  Created {len(all_chunks)} chunks from {len(documents)} files")
    return all_chunks


# ── Vector Store ───────────────────────────────────────────────────────────

def create_vector_store(chunks: List[Document]) -> FAISS:
    """Embed all code chunks and store in a FAISS vector index."""
    print(f"  Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"  Building FAISS index over {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"  Vector store ready!")
    return vector_store


# ── Helpers ────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a string for the prompt."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"--- File: {source} ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def print_sources(source_docs: List[Document]):
    """Pretty-print which files were used to answer the question."""
    seen = set()
    sources = []
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    if sources:
        print(f"\n  Sources: {', '.join(sources[:5])}")


# ── Interactive CLI ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert code analyst. Answer the user's question about their codebase based on the retrieved code snippets provided.

Rules:
- Reference specific file names and line details when possible.
- If the code snippets don't contain enough info, say so honestly.
- Keep answers clear and concise.
- When asked "where is X", point to the exact file(s) and describe the relevant code."""


def run_interactive(llm: HuggingFaceChatLLM, retriever):
    """Run an interactive Q&A loop in the terminal."""
    print("\n" + "=" * 60)
    print("  Code Agent Ready! Ask anything about the repo.")
    print("  Type 'quit' to exit, 'clear' to reset memory.")
    print("=" * 60 + "\n")

    chat_history: list = []

    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            chat_history.clear()
            print("  Memory cleared.\n")
            continue

        try:
            # Retrieve relevant code chunks
            docs = retriever.invoke(question)
            context = format_docs(docs)

            # Build the user message with context
            user_message = f"""Here are the relevant code snippets from the repository:

{context}

Question: {question}"""

            # Call the LLM
            answer = llm.chat(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
                chat_history=chat_history,
            )

            print(f"\nAgent > {answer}")
            print_sources(docs)
            print()

            # Update chat history (keep last 10 exchanges = 20 messages)
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except requests.exceptions.HTTPError as e:
            print(f"\n  API Error: {e}")
            print("  This might be a rate limit. Wait a moment and try again.\n")
        except Exception as e:
            print(f"\n  Error: {e}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path-to-repo>")
        print("Example: python agent.py ./my-project")
        sys.exit(1)

    repo_path = sys.argv[1]
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("Error: Set HUGGINGFACEHUB_API_TOKEN env variable.")
        print("Get a free token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    print("\nStep 1: Loading repository...")
    documents = load_repository(repo_path)
    if not documents:
        print("  No code files found. Check the path and try again.")
        sys.exit(1)

    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)

    print("\nStep 3: Building vector store...")
    vector_store = create_vector_store(chunks)

    print("\nStep 4: Setting up LLM...")
    llm = HuggingFaceChatLLM(model=LLM_MODEL, token=token)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    print(f"  Connected to {LLM_MODEL}")

    run_interactive(llm, retriever)


if __name__ == "__main__":
    main()