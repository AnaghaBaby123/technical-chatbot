# Pfeuffer RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot developed for technical machines. The chatbot enables users to ask questions about machine documentation and receive AI-generated answers based on the available manuals and documentation.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Multi-machine document support
- Section based chunking strategy
- Machine-based filtering
- PDF document ingestion
- Dockerized deployment
- Machine selection interface
- Multilingual query support
- Local Large Language Model (LLM)

---

## Tech Stack

- Python
- FastAPI
- LangChain
- Qdrant
- Docker & Docker Compose
- Nginx

---

## Project Structure

```
app/
│
├── backend/
├── frontend/
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

---

## Prerequisites

Before running the application, ensure you have:

- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Docker Compose
- NVIDIA GPU (recommended for local LLM inference)
- NVIDIA Container Toolkit (Linux)

---

## System Requirements

Before running the chatbot, ensure that your system has sufficient GPU memory. The required VRAM depends on the selected language model and deployment configuration.

| Component | Estimated VRAM Usage |
|-----------|---------------------:|
| Qwen3-30B-A3B (4-bit quantized, 14k-token context) | 20–25 GB |
| Qwen3-4B Reranker | 3–4 GB |
| BGE-M3 Embedding Model | 1 GB |
| Safety Margin | 2–4 GB |
| **Total Estimated VRAM** | **26–34 GB** |

**Note:** The first startup may take several minutes as the required AI models are downloaded and loaded into GPU memory. Systems with insufficient VRAM may experience slow performance or may be unable to run the application.

You could also change the components by changing their model name in the AdvancedRAGConfig class in maincode.py or changing in backendCode.py file when calling the main function.

---
## Adding  Machine Documentation

1. Create a new folder inside the backend/dataset directory  using the naming convention:

```
MachineName_AccessLevel
```
If access level is not needed, just leave it empty

2. Add the machine PDF files to this folder.

3. Also, update the machine list inside (Around code line 640)

```
frontend/app.py
```

by adding the machine name to the `MACHINES` list.

The machine name added here should be same as folder name. (case sensistiity is not a problem, but spelling should be same - as this name is used for machine-based filtering later)


---

## Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Navigate to the project folder.

```bash
cd <repository-name>/app
```

---

## Download the Language Identification Model

Navigate to the backend folder.

```bash
cd backend
```

Download the FastText language identification model.

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Return to the project root.

```bash
cd ..
```

---

## Running the Chatbot

Start all services using Docker Compose.

```bash
docker compose up
```

Or run in the background.

```bash
docker compose up -d
```

The first startup may take several minutes because the AI models are loaded into memory.

Once the backend has started successfully, open your browser and navigate to

```
http://localhost:8080
```

---

## Stopping the Application

Stop all containers.

```bash
docker compose down
```

---



