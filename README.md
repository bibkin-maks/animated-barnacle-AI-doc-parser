# 📄 PDF AI Reader – FastAPI Backend

Backend service for an AI-powered PDF assistant.  
Users sign in with Google, upload a PDF, and ask questions about its content.  
The service stores the parsed document and chat history in MongoDB and uses OpenAI to answer questions, optionally grounded in the uploaded PDF.

---

## ✨ Features

- 🔐 **Google Sign-In** using OAuth2 ID tokens (no password storage)
- 🔑 **JWT-based session auth** with per-user secret & logout invalidation
- 📄 **In-memory PDF processing** using `PyPDF2` (no files written to disk)
- 🧠 **Semantic search** over PDF chunks via FAISS + OpenAI embeddings
- 💬 **Persistent chat history** per user in MongoDB
- 🌐 **CORS configured** for a specific frontend origin (Vercel)
- 🧪 FastAPI with type-safe request models (Pydantic)

---

## 🧱 Tech Stack

- **Python** 3.11+
- **FastAPI** (API framework)
- **MongoDB** (user & document storage)
- **OpenAI API**
  - `gpt-4o-mini` for chat completions
  - `text-embedding-3-large` for embeddings
- **LangChain**
  - `FAISS` vector store
  - `CharacterTextSplitter` for PDF chunking
- **Google OAuth**
  - `google.oauth2.id_token` for token verification
- **JWT** via [`python-jose`](https://github.com/mpdavis/python-jose)
- **PyPDF2** for PDF text extraction

---

## 🏗️ Architecture Overview

1. **Auth flow**
   - Frontend obtains a Google ID token after user signs in.
   - Frontend sends that token to `POST /userauth/`.
   - Backend verifies the token with Google and either finds or creates a MongoDB user:
     - `_id` = Google `sub`
     - `jwt_secret` = per-user random hex string
   - Backend returns:
     - a signed JWT (`token`) for subsequent API calls
     - a sanitized `user` object with basic profile info.

2. **Document flow**
   - Authenticated user calls `POST /upload/` with a PDF (`multipart/form-data`).
   - Backend:
     - Reads the file into memory
     - Extracts text from all pages via `PyPDF2`
     - Splits text into chunks using `CharacterTextSplitter`
     - Stores `{ name, chunks }` inside the user document in MongoDB.

3. **Query flow**
   - Authenticated user sends a question to `POST /query/`.
   - Backend:
     - Verifies the JWT and loads the user’s document.
     - If a PDF exists:
       - Builds a FAISS vector store from stored chunks.
       - Finds top-k relevant chunks via similarity search.
       - Calls OpenAI with a system prompt telling it to answer **only** from those chunks.
     - If no PDF exists:
       - Calls OpenAI with a generic assistant prompt.
     - Saves the conversation (user question + AI answer) in MongoDB.
     - Returns: `{ answer, matched_chunks, document_used }`.

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
