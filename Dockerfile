# Use an official Python base image
FROM python:3.12.7-slim

# Set the working directory
WORKDIR /app

# Copy your code
COPY main.py data_extractor.py ./

# Copy optional folders (if used)
# COPY ./chroma ./chroma
# COPY ./data_mail_extract ./data_mail_extract

# Install system dependencies (optional: poppler-utils, libmagic for some loaders)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libmagic1 \
    poppler-utils \
    libglib2.0-0 \
    libcairo2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
    
RUN pip install --upgrade pip
  
# Install Python dependencies
RUN pip install --no-cache-dir \
    six \
    "urllib3<2.0" \
    fastapi==0.115.5 \
    uvicorn==0.32.0 \
    langchain==0.3.7 \
    langchain-community==0.3.7 \
    langchain-text-splitters==0.3.2 \
    langchain-ollama==0.3.3 \
    chromadb==1.0.15 \
    pypdf==5.1.0 \
    unstructured==0.18.1 \
    python-multipart==0.0.20 \
    requests==2.21.0 \
    transformers==4.41.1 \
    PyPDF2==3.0.1 \
    rank-bm25==0.2.2 \
    faiss-cpu==1.11.0 \
    pdfplumber==0.11.7 \
    pdfminer.six==20250506 \
    PyCryptodome==3.23.0 \
    nltk==3.9.1 \
    sentence-transformers==3.3.0
    

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
