# Gen AI with LangChain

A comprehensive collection of Generative AI projects and examples built with LangChain, demonstrating various LLM integration patterns, RAG pipelines, and AI application development techniques.

## Project Overview

This repository contains practical implementations and experiments with LangChain framework, covering everything from basic LLM interactions to advanced RAG (Retrieval-Augmented Generation) systems.

## Directory Structure

```
GEN AI WITH LANGCHAIN/
├── CHAT MODEL/              # Chat model implementations
│   ├── chat_model.py        # HuggingFace chat model
│   ├── chat_model_local.py  # Local LLM (TinyLlama)
│   └── google_chat_model.py # Google Gemini integration
├── chain/                   # Chain implementations
│   ├── conditional_chain.py # Conditional/branching chains
│   └── prallelchain.py      # Parallel chain execution
├── chatbot/                 # Basic chatbot implementation
├── dynamic chat template/   # Dynamic prompt templating
├── EMBEDING/                # Embedding examples
│   ├── doc_similarity.py    # Document similarity with cosine
│   └── embeding_query.py    # Query embedding
├── langchain_structured_output/ # Structured output with Pydantic
├── LLM/                     # Basic LLM examples
├── messages type/           # Message handling (System, Human, AI)
├── outparses/               # Output parsers
│   ├── json_output_parser.py
│   ├── pydantic_parser.py
│   ├── structure_output_parser.py
│   └── sting_output_parses.py
├── prompts/                 # Prompt engineering
│   ├── prompt.py            # Basic prompts
│   └── prompt_gene.py       # Advanced prompt templates
├── RAG/                     # Retrieval-Augmented Generation
│   ├── document_loader/     # Document loading utilities
│   │   ├── pdf_loader.py
│   │   ├── text_loader.py
│   │   └── web_bassedloader.py
│   ├── text spliters/       # Text splitting strategies
│   │   ├── length_based.py
│   │   ├── sementic_based.py
│   │   └── document_struture_based.py
│   ├── retrevers/           # Retrieval strategies
│   │   ├── contextual_retrievers.ipynb
│   │   ├── MMR_retrivels.ipynb
│   │   ├── multi_query_retrievers.ipynb
│   │   └── vector_store_retriever.ipynb
│   └── vector store/        # Vector database implementations
│       ├── fasii_langchain.ipynb
│       └── langchain_chromadb.ipynb
├── RUNNABLES/               # LangChain Runnables (LCEL)
│   ├── runnable_lamda.py
│   ├── premitive_runable_parralle.py
│   ├── premietive_runable_sequence.py
│   └── runable_abstract.py
├── youtube_chatbot/         # YouTube transcript chatbot
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── template.json            # Prompt template configuration
└── my_prompt.json           # Custom prompt configuration
```

## Features & Components

### 1. Chat Models
- **HuggingFace Models**: Integration with Meta-Llama, DeepSeek, Gemma models
- **Google Gemini**: ChatGoogleGenerativeAI integration
- **Groq**: Fast inference with Llama-3.3-70B
- **Local LLM**: TinyLlama-1.1B for offline usage

### 2. Chains
- **Conditional Chains**: Branch logic based on sentiment classification
- **Parallel Chains**: Execute multiple chains simultaneously and merge results
- **Runnable Sequences**: Chain composition with LCEL (LangChain Expression Language)

### 3. Output Parsers
- **StrOutputParser**: Simple string output parsing
- **JsonOutputParser**: Parse LLM output as JSON
- **PydanticOutputParser**: Structured output with Pydantic validation
- **StructuredOutputParser**: Schema-based output with response schemas

### 4. Prompts
- **Prompt Templates**: Reusable prompt templates with variables
- **Dynamic Prompts**: Runtime prompt construction
- **Advanced Templates**: Academic synthesis engine with style-driven projection

### 5. RAG (Retrieval-Augmented Generation)
- **Document Loaders**:
  - PDF documents (PyPDFLoader)
  - Text files (TextLoader)
  - Web pages (WebBaseLoader)
  
- **Text Splitters**:
  - Character-based splitting
  - Length-based splitting
  - Semantic chunking (meaning-based)
  
- **Vector Stores**:
  - FAISS (Facebook AI Similarity Search)
  - ChromaDB (persistent vector store)
  
- **Retrievers**:
  - Basic similarity search
  - MMR (Maximum Marginal Relevance)
  - Multi-query retrieval
  - Contextual compression retrieval

### 6. Embeddings
- HuggingFace sentence-transformers
- Document similarity with cosine similarity
- Query embedding for semantic search

### 7. Runnables (LCEL)
- RunnableSequence: Chain components sequentially
- RunnableParallel: Execute multiple chains in parallel
- RunnableLambda: Custom transformations
- RunnablePassthrough: Pass data through chains
- Custom abstract runnable implementation

### 8. YouTube Chatbot
- YouTube transcript fetching
- Multi-language support (English/Hindi)
- RAG pipeline setup for video content

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd "D:\GEN AI WITH LANGCHAIN"
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
GROQ_API_KEY=your_key_here
```

## Dependencies

| Package | Purpose |
|---------|---------|
| langchain | Core LangChain framework |
| langchain-core | LangChain core abstractions |
| langchain-openai | OpenAI integration |
| langchain-anthropic | Anthropic Claude integration |
| langchain-google-genai | Google Gemini integration |
| langchain-huggingface | HuggingFace models & embeddings |
| langchain-community | Community integrations (FAISS, Chroma) |
| openai | OpenAI API client |
| google-generativeai | Google AI client |
| transformers | HuggingFace transformers |
| huggingface-hub | HuggingFace Hub client |
| python-dotenv | Environment variable management |
| numpy | Numerical operations |
| scikit-learn | ML utilities (cosine similarity) |

## Usage Examples

### Basic Chat Model
```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!")
]
response = model.invoke(messages)
```

### RAG Pipeline
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load document
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Query
results = vectorstore.similarity_search("your query here")
```

### Chain with Output Parser
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace

parser = JsonOutputParser()
template = PromptTemplate(
    template="Generate data about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "Harry Potter"})
```

## Key Concepts Demonstrated

1. **LCEL (LangChain Expression Language)**: Modern chain composition using `|` operator
2. **Structured Output**: Pydantic-based output validation and parsing
3. **Semantic Search**: Vector embeddings for meaning-based retrieval
4. **Contextual Compression**: Improve retrieval quality with LLM-based compression
5. **Multi-Model Pipelines**: Combine different LLMs (HuggingFace + Groq) in single chains
6. **Prompt Engineering**: Advanced prompt templates with role definition and constraints

## Project Highlights

### Conditional Chain (`chain/conditional_chain.py`)
Demonstrates sentiment-based routing:
- Classifies feedback as positive/negative
- Routes to appropriate response chain
- Uses different models for different tasks

### Parallel Chain (`chain/prallelchain.py`)
Shows parallel execution:
- Generates notes and Q&A simultaneously
- Merges results into cohesive output

### Contextual Retriever (`RAG/retrevers/contextual_retrievers.ipynb`)
Advanced RAG technique:
- Retrieves documents with LLM-based compression
- Extracts only relevant portions from documents
- Improves answer quality while reducing token usage

## License

Educational/Research purposes

## Acknowledgments

- LangChain team for the amazing framework
- HuggingFace for open-source models
- Google, Anthropic, OpenAI for LLM APIs
