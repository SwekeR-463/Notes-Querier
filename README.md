# Notes Querier

This project allows you to upload your notes in PDF format and ask questions about the content. The system uses natural language processing and retrieval-based question answering to help you study efficiently.

## Features

- **PDF Upload**: Easily upload your notes in PDF.
- **Question Answering**: Ask questions and get concise answers.
- **Interactive Interface**: Simple, user-friendly web interface using Gradio.

## Tech Stack

- **LangChain**: For LLM Orchestration.
- **Gradio**: For the interactive user interface.
- **Ollama**: For running models locally.
- **FAISS**: For fast and efficient document retrieval.
- **PyMuPDF**: For processing and extracting text from PDFs.

## Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/SwekeR-463/Notes-Querier.git
    cd notes-qa-assistant
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install Ollama**

    - **Windows**: Download and install from [Ollama's website](https://ollama.com/download).
    - **macOS**: Install via Homebrew:
    
      ```bash
      brew install ollama/tap/ollama
      ```

5. **Pull Required Models**

    ```bash
    ollama pull nomic-embed-text
    ollama pull gemma2:2b
    ```
    Feel free to use any model you like based upon your system.

## Usage

1. **Run the Application**

    ```bash
    python rag.py
    ```

2. **Interact with the Interface**

    - Upload your notes in PDF.
    - Ask your questions and get answers.
