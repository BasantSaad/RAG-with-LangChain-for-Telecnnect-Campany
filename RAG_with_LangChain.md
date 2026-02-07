# RAG with LangChain: Building an Intelligent Q&A Chatbot

**Presented by:**  
Basant Saad Eldin Mohammed  
Mena Gamal Aboelela

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is RAG?](#what-is-rag)
3. [Project Overview](#project-overview)
4. [Technology Stack](#technology-stack)
5. [Setup and Installation](#setup-and-installation)
6. [Implementation Details](#implementation-details)
   - [Data Loading](#data-loading)
   - [Text Chunking](#text-chunking)
   - [Vector Embeddings and FAISS](#vector-embeddings-and-faiss)
   - [LLM Integration](#llm-integration)
   - [Query Processing](#query-processing)
   - [Interactive GUI](#interactive-gui)
7. [Demo Results](#demo-results)
8. [Key Takeaways](#key-takeaways)
9. [Code Reference](#code-reference)

---

## Introduction

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using LangChain to build an intelligent question-answering chatbot. The chatbot is designed to answer customer service queries based on TeleConnect's knowledge base, combining document retrieval with large language model generation for accurate, context-aware responses.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI framework that combines the power of retrieval systems with generative AI models to provide accurate, context-aware responses.

### Key Benefits:

- **Reduces Hallucinations**: By grounding responses in retrieved documents, RAG ensures answers are based on actual data rather than model imagination
- **Access to Current Information**: Enables AI to access up-to-date information beyond its training data
- **Cost-Effective**: Provides an alternative to expensive fine-tuning of large language models
- **Flexibility**: Easy to update knowledge base without retraining the model

### How RAG Works:

1. **Documents** are loaded and processed
2. **Retrieval** system finds relevant information
3. **Generation** model creates answers based on retrieved context

---

## Project Overview

### Goal

Build an intelligent chatbot using RAG that answers questions based on TeleConnect's customer service knowledge base.

### Components

The system consists of five main components:

1. **Document Loading**: Processing PDF and CSV files containing service information
2. **Text Chunking**: Breaking documents into manageable, semantically meaningful pieces
3. **Vector Database**: Using FAISS for efficient similarity search
4. **LLM Integration**: Leveraging Mistral-Nemo for natural language response generation
5. **Interactive GUI**: Providing a user-friendly interface built with ipywidgets

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| **LangChain** | Framework for building LLM applications |
| **FAISS** | Fast vector similarity search and clustering |
| **HuggingFace** | Providing embeddings and LLM models |
| **PyPDF** | Loading and parsing PDF documents |
| **Transformers** | Interface for working with transformer models |
| **ipywidgets** | Creating interactive GUI components |

### Core Dependencies

- **FAISS** (Facebook AI Similarity Search): Enables efficient vector storage and retrieval
- **LangChain**: Provides tools for document processing and RAG pipeline construction
- **Sentence-Transformers**: Generates high-quality text embeddings
- **PyTorch & Transformers**: Powers the large language model

---

## Setup and Installation

### Required Libraries

Install all necessary dependencies using pip:

```bash
pip install faiss-cpu langchain \
  langchain-community langchain-core \
  pypdf sentence-transformers \
  transformers==4.52.4 torch
```

### Import Statements

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ipywidgets as widgets
from IPython.display import display
```

---

## Implementation Details

### Data Loading

The system supports loading documents from multiple file formats:

#### PDF Files

The `PyPDFLoader` extracts text from PDF documents:

```python
loader = PyPDFLoader(file_path)
all_documents.extend(loader.load())
```

#### CSV Files

The `CSVLoader` handles structured data like service plans:

```python
loader = CSVLoader(file_path=file_path, encoding="utf-8")
all_documents.extend(loader.load())
```

#### Multi-Source Loading

The implementation supports loading from multiple files:

```python
files_to_load = [
    "/content/TeleConnect_Customer_Service_Knowledge_Base.pdf",
    "/content/TeleConnect_Service_Plans.csv"
]

all_documents = []

for file_path in files_to_load:
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        continue
    
    print(f"Loading: {file_path}...")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())
    
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        all_documents.extend(loader.load())
```

---

### Text Chunking

#### Why Chunking?

Documents must be split into smaller chunks for several reasons:

- **Context Window Limits**: LLMs have maximum input sizes
- **Retrieval Precision**: Smaller chunks enable more precise information retrieval
- **Embedding Quality**: Optimal chunk sizes improve embedding effectiveness
- **Memory Efficiency**: Reduces computational overhead

#### Chunking Configuration

```python
text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # Maximum characters per chunk
    chunk_overlap=100     # Overlap between consecutive chunks
)

chunks = text_splitter.split_documents(all_documents)
```

#### Benefits of Overlap

The 100-character overlap between chunks:
- Maintains context across chunk boundaries
- Prevents loss of information at split points
- Improves retrieval quality for queries spanning multiple chunks

**Result**: The system created 4 chunks from the loaded documents.

---

### Vector Embeddings and FAISS

#### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2`, a lightweight but powerful embedding model:

```python
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'}  # CPU-based to save GPU memory
)
```

#### Vector Database Creation

FAISS indexes the embedded chunks for fast similarity search:

```python
vectordb = FAISS.from_documents(chunks, embedding)
```

#### How It Works

1. Each text chunk is converted into a high-dimensional vector (embedding)
2. These vectors capture the semantic meaning of the text
3. FAISS indexes the vectors using efficient data structures
4. During queries, user questions are embedded using the same model
5. FAISS finds the most semantically similar document chunks

#### Why FAISS?

- **Speed**: Extremely fast similarity search, even for large datasets
- **Efficiency**: Optimized memory usage and computational performance
- **Scalability**: Industry-standard for production vector databases
- **Flexibility**: Supports various distance metrics and indexing methods

---

### LLM Integration

#### Model Selection

The project uses **Mistral-Nemo-Instruct-2407**, a powerful instruction-tuned language model:

```python
model_name = "mistralai/Mistral-Nemo-Instruct-2407"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Half precision for memory efficiency
    device_map="auto"            # Automatic device placement
)
```

#### Text Generation Function

```python
def generate_text(prompt, max_length=512, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
```

#### Generation Parameters

- **max_length**: 512 tokens (reduced to 256 in chatbot to prevent memory issues)
- **temperature**: 0.7 for balanced creativity vs. accuracy
- **top_k**: 50 for quality control
- **top_p**: 0.95 for nucleus sampling
- **do_sample**: True for diverse, natural responses

#### Memory Optimization

- **torch_dtype=float16**: Reduces memory footprint by 50%
- **device_map="auto"**: Optimal GPU/CPU distribution
- **CPU-based embeddings**: Conserves GPU memory for model inference

---

### Query Processing

The query processing flow combines retrieval and generation:

```python
def ask_question(query):
    # Step 1: Retrieve relevant documents
    docs = vectordb.similarity_search(query, k=1)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 2: Create prompt with context
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
    
    # Step 3: Generate response
    result = generate_text(prompt, max_length=256)[0]
    return result.strip()
```

#### Processing Steps

1. **User submits question** via the GUI
2. **Question is embedded** into a vector representation
3. **FAISS finds** the most similar document chunk (k=1)
4. **Context and question** are combined into a structured prompt
5. **LLM generates** an answer based on the provided context
6. **Response is displayed** in the output area

#### Prompt Engineering

The prompt structure includes:
- **System instruction**: Defines the assistant's role
- **Context**: Retrieved document chunk
- **Question**: User's query
- **Answer trigger**: Prompts the model to generate a response

---

### Interactive GUI

#### GUI Components

The chatbot interface uses `ipywidgets` for interactive elements:

```python
# Text input area
text_input = widgets.Textarea(
    value='',
    placeholder='Type your question here...',
    description='Your Question:',
    disabled=False,
    layout=widgets.Layout(width='700px', height='auto')
)

# Output display area
output_area = widgets.Output()

# Submit button
ask_button = widgets.Button(
    description='Ask',
    disabled=False,
    button_style='',
    tooltip='Click to ask the question',
    icon='comment'
)
```

#### Event Handler

The button click event processes user queries:

```python
def on_button_click(button):
    with output_area:
        output_area.clear_output()
        user_question = text_input.value
        print(f"You asked: {user_question}")
        
        if user_question:
            try:
                answer = ask_question(user_question)
                print(f"Chatbot: {answer}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Please type a question.")
        
        text_input.value = ''

ask_button.on_click(on_button_click)
```

#### Display Layout

The components are arranged in a vertical box:

```python
display(widgets.VBox([text_input, ask_button, output_area]))
```

#### GUI Features

- **Automatic input clearing**: Enhances user experience for multiple queries
- **Error handling**: Graceful failure management with informative messages
- **Clean layout**: Intuitive VBox container organization
- **Real-time feedback**: Immediate display of questions and answers

---

## Demo Results

### Example Interaction

**User Query:**
```
"How much is the price of basic?"
```

**Chatbot Response:**
```
"The monthly price of the Basic plan is $25."
```

### System Performance

- ✅ Successfully loaded 4 document chunks
- ✅ Accurate retrieval from CSV service plan data
- ✅ Concise, context-grounded response generation
- ✅ No hallucination - answer directly from knowledge base
- ✅ Fast response time with optimized settings

### Technical Achievements

1. **Multi-format support**: Successfully integrated PDF and CSV data sources
2. **Efficient retrieval**: FAISS quickly identified the relevant chunk
3. **Context-aware generation**: LLM provided accurate answer based on retrieved data
4. **User-friendly interface**: Clean, functional GUI for easy interaction

---

## Key Takeaways

1. **RAG combines retrieval and generation** for accurate, grounded responses that minimize hallucinations

2. **LangChain simplifies** building complex LLM applications with modular, reusable components

3. **FAISS enables efficient similarity search** at scale, making it ideal for production systems

4. **Proper chunking and overlap** maintain context integrity and improve retrieval quality

5. **Memory optimization is crucial** for running large models, especially with limited resources

6. **Interactive GUIs** make AI accessible to end users without technical expertise

7. **Context-grounding prevents hallucinations**, ensuring responses are based on actual data

8. **Modular architecture** allows easy updates to knowledge base without retraining models

---

## Code Reference

### Complete Implementation

#### 1. Installation

```bash
!pip install faiss-cpu langchain langchain-community langchain-core pypdf sentence-transformers transformers==4.52.4 torch
```

#### 2. Imports

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ipywidgets as widgets
from IPython.display import display
import os
```

#### 3. Load Model

```python
model_name = "mistralai/Mistral-Nemo-Instruct-2407"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

#### 4. Define Generation Function

```python
def generate_text(prompt, max_length=512, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
```

#### 5. Load and Process Documents

```python
files_to_load = [
    "/content/TeleConnect_Customer_Service_Knowledge_Base.pdf",
    "/content/TeleConnect_Service_Plans.csv"
]

all_documents = []

for file_path in files_to_load:
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        continue
    
    print(f"Loading: {file_path}...")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())
    
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        all_documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(all_documents)

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'}
)
vectordb = FAISS.from_documents(chunks, embedding)

print(f"\nSuccess! Total chunks in knowledge base: {len(chunks)}")
```

#### 6. Define Query Function

```python
def ask_question(query):
    docs = vectordb.similarity_search(query, k=1)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
    
    result = generate_text(prompt, max_length=256)[0]
    return result.strip()
```

#### 7. Create GUI

```python
text_input = widgets.Textarea(
    value='',
    placeholder='Type your question here...',
    description='Your Question:',
    disabled=False,
    layout=widgets.Layout(width='700px', height='auto')
)

output_area = widgets.Output()

ask_button = widgets.Button(
    description='Ask',
    disabled=False,
    button_style='',
    tooltip='Click to ask the question',
    icon='comment'
)

def on_button_click(button):
    with output_area:
        output_area.clear_output()
        user_question = text_input.value
        print(f"You asked: {user_question}")
        
        if user_question:
            try:
                answer = ask_question(user_question)
                print(f"Chatbot: {answer}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Please type a question.")
        
        text_input.value = ''

ask_button.on_click(on_button_click)

display(widgets.VBox([text_input, ask_button, output_area]))
```

---

## Conclusion

This project successfully demonstrates the power of RAG in building intelligent, context-aware chatbots. By combining document retrieval with large language models, we created a system that provides accurate answers grounded in actual data, avoiding the hallucination problems common in standalone LLMs.

The modular architecture allows for easy expansion and updates to the knowledge base, making it a practical solution for real-world customer service applications.

---

**Project Contributors:**
- Basant Saad Eldin Mohammed
- Mena Gamal Aboelela
