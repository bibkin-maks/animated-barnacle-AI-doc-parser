from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import id_token
from google.auth.transport import requests
from jose import jwt
from datetime import datetime, timedelta, timezone
import urllib.parse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from secrets import token_hex

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def empty_document():
    """Return the unified empty-document structure."""
    return {"name": "empty file", "chunks": ["Nothing here yet"]}

def is_document_empty(doc):
    """Correctly detect an empty file in ALL formats."""
    if not doc:
        return True

    name = doc.get("name", "")
    chunks = doc.get("chunks", "")

    if name in ["", None, "empty file"]:
        return True

    # chunks is list of empty message
    if isinstance(chunks, list) and (
        len(chunks) == 0 or chunks[0] == "Nothing here yet"
    ):
        return True

    # chunks is empty string
    if isinstance(chunks, str) and chunks.strip() == "":
        return True

    return False

# -------------------------------------------------------------------
# Load env & Mongo
# -------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MONGODB_URI_FIRST = os.getenv("MONGODB_URI_FIRST")
MONGODB_URI_PASS = os.getenv("MONGODB_URI_PASS")
MONGODB_URI_LAST = os.getenv("MONGODB_URI_LAST")

MONGODB_URI = f"{MONGODB_URI_FIRST}{urllib.parse.quote(MONGODB_URI_PASS)}{MONGODB_URI_LAST}"

mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = mongo_client["pdfAIReader"]
users_collection = db["users"]

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://animated-barnacle-ai-doc-parser-fro.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# -------------------------------------------------------------------
# Google Token Verification
# -------------------------------------------------------------------

def verify_google_token(token):
    try:
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID
        )
        return {
            "sub": idinfo["sub"],
            "email": idinfo["email"],
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }
    except ValueError:
        return None

# -------------------------------------------------------------------
# JWT
# -------------------------------------------------------------------

def create_jwt(user: dict):
    signing_key = SECRET_KEY + user["jwt_secret"]

    payload = {
        "user_id": user["_id"],
        "email": user["email"],
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, signing_key, algorithm="HS256")

def verify_jwt(token: str):
    unverified = jwt.get_unverified_claims(token)
    user_id = unverified.get("user_id")
    if not user_id:
        raise HTTPException(401, "Invalid token")

    user = users_collection.find_one({"_id": user_id})
    if not user:
        raise HTTPException(401, "User not found")

    signing_key = SECRET_KEY + user["jwt_secret"]

    try:
        jwt.decode(token, signing_key, algorithms=["HS256"])
        return user
    except:
        raise HTTPException(401, "Token invalid or expired")

# -------------------------------------------------------------------
# User Management
# -------------------------------------------------------------------

def get_or_create_user(idinfo):
    user_id = idinfo["sub"]
    existing = users_collection.find_one({"_id": user_id})
    if existing:
        return existing

    new_user = {
        "_id": user_id,
        "email": idinfo["email"],
        "name": idinfo.get("name"),
        "picture": idinfo.get("picture"),
        "messages": [
            {"role": "AI", "content": f"Hello {idinfo.get('name')}! How can I assist you?"}
        ],
        "jwt_secret": token_hex(32),
        "document": empty_document()
    }

    users_collection.insert_one(new_user)
    return new_user

# -------------------------------------------------------------------
# Chat Message Handling
# -------------------------------------------------------------------

def add_message_to_user(user, role, content):
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$push": {"messages": {"role": role, "content": content}}}
    )

# -------------------------------------------------------------------
# Query Function
# -------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

def query(question: str, token: str):
    user = verify_jwt(token)
    user_doc = user.get("document")

    if is_document_empty(user_doc):
        raise HTTPException(400, "You must upload a non-empty PDF first.")

    chunks = user_doc["chunks"]
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    docs = vector_store.similarity_search(question, k=3)

    context = "\n".join([d.page_content for d in docs])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer ONLY using the context."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ]
    )

    ai_text = response.choices[0].message.content

    add_message_to_user(user, "user", question)
    add_message_to_user(user, "AI", ai_text)

    return {
        "answer": ai_text,
        "matched_chunks": len(docs)
    }

# -------------------------------------------------------------------
# PDF Processing
# -------------------------------------------------------------------

def process_document(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks]

# -------------------------------------------------------------------
# Upload Document
# -------------------------------------------------------------------

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...), authorization: str = Header(None)):
    user = verify_jwt(authorization.replace("Bearer ", ""))

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed.")

    save_path = os.path.join("uploaded_documents", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_document(save_path)

    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"document": {"name": file.filename, "chunks": chunks}}}
    )

    return {"message": "Document uploaded."}

# -------------------------------------------------------------------
# Remove Document
# -------------------------------------------------------------------

@app.post("/removefile/")
async def delete_document(authorization: str = Header(None)):
    user = verify_jwt(authorization.replace("Bearer ", ""))
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"document": empty_document()}}
    )
    return {"message": "Document deleted."}

# -------------------------------------------------------------------
# Query Endpoint
# -------------------------------------------------------------------

@app.post("/query/")
async def ask_question(request: QueryRequest, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    return query(request.question, token)

# -------------------------------------------------------------------
# Auth
# -------------------------------------------------------------------

@app.post("/userauth/")
async def user_auth(user: dict):
    idinfo = verify_google_token(user.get("token"))
    if not idinfo:
        raise HTTPException(401, "Invalid Google token")

    user_record = get_or_create_user(idinfo)
    jwt_token = create_jwt(user_record)
    return {"token": jwt_token, "user": user_record}

# -------------------------------------------------------------------
# Signout
# -------------------------------------------------------------------

@app.post("/signout/")
async def user_signout(token: str):
    user = verify_jwt(token)
    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"jwt_secret": token_hex(32)}}
    )
    return {"message": "Signed out"}

# -------------------------------------------------------------------
# Me
# -------------------------------------------------------------------

@app.get("/me")
async def get_current_user(authorization: str = Header(None)):
    token = authorization.replace("Bearer ", "")
    user = verify_jwt(token)
    fresh = users_collection.find_one({"_id": user["_id"]})
    return fresh

# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
