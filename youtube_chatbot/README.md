# YouTube Chatbot

A LangChain-based chatbot that enables conversational interaction with YouTube video content by fetching and processing video transcripts.

## Overview

This project demonstrates how to build a RAG (Retrieval-Augmented Generation) chatbot that can answer questions about YouTube videos. It fetches video transcripts, processes them into chunks, creates embeddings, and stores them in a FAISS vector store for efficient retrieval.

## Features

- **YouTube Transcript Fetching**: Automatically fetches transcripts from YouTube videos
- **Multi-language Support**: Attempts to fetch English transcripts first, with fallback to Hindi
- **Text Processing**: Splits transcripts into manageable chunks using LangChain's RecursiveCharacterTextSplitter
- **Vector Embeddings**: Uses HuggingFace embeddings for semantic understanding
- **Vector Store**: FAISS for efficient similarity search and retrieval

## Installation

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Setup

1. Clone or navigate to the project directory:
   ```bash
   cd youtube_chatbot
   ```

2. Install required dependencies:
   ```bash
   pip install youtube-transcript-api
   pip install langchain-text-splitters langchain-core langchain-huggingface langchain-community faiss-cpu
   ```

3. Run the notebook:
   ```bash
   jupyter notebook chat_bot.ipynb
   ```

## Usage

### Basic Usage

1. Open `chat_bot.ipynb` in Jupyter Notebook
2. Run the installation cell to install `youtube-transcript-api`
3. Import the required libraries
4. Set the YouTube video ID (extracted from the video URL)
5. Run the transcript fetching cell

### Extracting Video ID

From a YouTube URL like `https://www.youtube.com/watch?v=J5_-l7WIO_w`, the video ID is `J5_-l7WIO_w`.

### Code Example

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

video_id = "J5_-l7WIO_w"  # Replace with your video ID
api = YouTubeTranscriptApi()

# Fetch transcript with language fallback
try:
    transcript_list = api.fetch(video_id, languages=['en'])
    transcript_text = " ".join([snippet.text for snippet in transcript_list.snippets])
    print("English transcript fetched successfully!")
except NoTranscriptFound:
    # Fallback to Hindi
    transcript_list = api.fetch(video_id, languages=['hi'])
    transcript_text = " ".join([snippet.text for snippet in transcript_list.snippets])
    print("Hindi transcript fetched successfully!")
```

## Project Structure

```
youtube_chatbot/
├── chat_bot.ipynb          # Jupyter notebook with transcript fetching implementation
└── README.md               # Project documentation
```

## Notebook Contents

The `chat_bot.ipynb` notebook contains:

1. **Installation**: Installs `youtube-transcript-api` package
2. **Imports**: Sets up LangChain components for RAG pipeline
   - `RecursiveCharacterTextSplitter` - for chunking transcripts
   - `HuggingFaceEmbeddings` - for creating vector embeddings
   - `FAISS` - for vector similarity search
3. **Transcript Fetching**: Fetches YouTube video transcripts with multi-language support (English → Hindi fallback)

## Dependencies

| Package | Purpose |
|---------|---------|
| youtube-transcript-api | Fetch YouTube video transcripts |
| langchain-text-splitters | Split text into chunks |
| langchain-core | Core LangChain abstractions |
| langchain-huggingface | HuggingFace embeddings |
| langchain-community | Community-maintained LangChain components |
| faiss-cpu | Vector similarity search |

## Limitations

- Only works with videos that have captions/transcripts enabled
- Some videos may not have transcripts available in the requested languages
- Transcript quality depends on whether it's auto-generated or manually created

## Future Enhancements

- [ ] Add LLM integration for question-answering
- [ ] Support for multiple videos in a single chatbot
- [ ] Add a Streamlit/Gradio UI for interactive chatting
- [ ] Implement conversation memory
- [ ] Add support for more languages

## License

This project is for educational purposes as part of learning LangChain and Generative AI.
