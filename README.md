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

## Adding New Machine Documentation

1. Create a new folder inside the dataset directory using the naming convention:

```
MachineName AccessLevel
```
If access levl is not needed, just leave it empty

2. Add the machine PDF files to this folder.

3. Update the machine list inside

```
frontend/app.py
```

by adding the machine name to the `MACHINES` list.

4. Rebuild the application.

```bash
docker compose build
docker compose up
```

---



