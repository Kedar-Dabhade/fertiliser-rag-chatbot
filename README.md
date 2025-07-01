# Fertiliser RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot for fertiliser product recommendations and Q&A, powered by OpenAI and OpenSearch.

---

## Features

- Conversational AI for fertiliser product queries
- Contextual retrieval using OpenSearch vector search
- Modern React-based frontend
- Session-based chat history

---

## Prerequisites

- **Python 3.8+**
- **Node.js 16+** and **npm**
- **Docker** (for running OpenSearch locally)
- **OpenAI API Key**

---

## Quick Start

### 1. Clone the Repository

```sh
git clone https://github.com/Kedar-Dabhade/fertiliser-rag-chatbot.git
cd fertiliser-rag-chatbot
```

---

### 2. Set Up OpenSearch (Recommended: Docker)

**a. Start OpenSearch using Docker Compose:**

```sh
docker-compose up -d
```

This will start OpenSearch and Dashboard on the default ports (9200, 5601).

**b. (Optional) Manual Docker Run:**

```sh
docker run -d --name opensearch -p 9200:9200 -e "discovery.type=single-node" -e "plugins.security.disabled=true" opensearchproject/opensearch:2.11.1
```

---

### 3. Backend Setup

**a. Create and activate a virtual environment:**

```sh
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

**b. Install Python dependencies:**

```sh
pip install -r requirements.txt
```

**c. Create a `.env` file in the project root:**

```env
OPENAI_API_KEY=your-openai-key
OPENSEARCH_HOST=http://localhost:9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin
OPENSEARCH_INDEX=opensearch_fert
OPENSEARCH_VERIFY_CERTS=false
SIMILARITY_THRESHOLD=0.75
MAX_CONTEXT_DOCS=3
K_CANDIDATES=7
```

**d. (Optional) Ingest Data:**

> **You only need to run the ingestion step if you are setting up OpenSearch for the first time, or if you want to update the product data.**
>
> If the OpenSearch index already contains the required data, you can skip this step.

To ingest product data:

```sh
python ingestos.py
```

---

### 4. Frontend Setup

```sh
cd fert-chatbot-ui
npm install
npm start
```

The frontend will run on [http://localhost:3000](http://localhost:3000).

---

### 5. Running the Backend API

In the project root:

```sh
python fertiliser_chatbot_api_opensearch.py
```

The API will be available at [http://localhost:5000](http://localhost:5000).

---

## OpenSearch Configuration

- The default configuration uses Docker and the credentials in your `.env`.
- If you use a remote OpenSearch instance, update `OPENSEARCH_HOST`, `OPENSEARCH_USER`, and `OPENSEARCH_PASSWORD` in your `.env`.
- The backend will create the index and mappings automatically if they do not exist.

---

## Environment Variables

All sensitive and environment-specific settings are managed via the `.env` file.  
**Never commit your `.env` to version control.**

---

## Troubleshooting

- **OpenSearch not running?**  
  Check Docker status: `docker ps`  
  Logs: `docker logs opensearch`
- **API errors?**  
  Ensure `.env` is configured and OpenSearch is running.
- **Frontend not connecting?**  
  Make sure the backend is running on port 5000 and CORS is enabled.

---

## Folder Structure

```
fertiliser-rag-chatbot/
│
├── fertiliser_chatbot_api_opensearch.py   # Main backend API
├── ingestos.py                            # Data ingestion script (run only if you need to ingest/update data)
├── requirements.txt
├── .env (not committed)
├── fert-chatbot-ui/                       # React frontend
│   ├── src/
│   ├── public/
│   └── ...
├── docker-compose.yaml                    # For OpenSearch
└── ...
```

---

## Production Notes

- Change all default passwords and secrets before deploying.
- Use HTTPS and secure your OpenSearch instance in production.
- Monitor API usage and logs for errors or abuse. 