from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # ← OPENAI EMBEDDINGS
from langchain_text_splitters import CharacterTextSplitter
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import id_token
from google.auth.transport import requests
from jose import jwt
from datetime import datetime, timedelta, timezone
from fastapi.middleware.cors import CORSMiddleware
import urllib.parse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from secrets import token_hex



# -------------------------------------------------------------------
# Environment & API Client
# -------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


MONGODB_URI_FIRST = os.getenv("MONGODB_URI_FIRST")
MONGODB_URI_PASS = os.getenv("MONGODB_URI_PASS")
MONGODB_URI_LAST = os.getenv("MONGODB_URI_LAST")

MONGODB_URI = f"{MONGODB_URI_FIRST}{urllib.parse.quote(MONGODB_URI_PASS)}{MONGODB_URI_LAST}"

print("Connecting to MongoDB with URI:", MONGODB_URI)


mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

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
# Constants
# -------------------------------------------------------------------
VECTOR_STORE_PATH = "faiss_vector_store"
UPLOAD_FOLDER = "uploaded_documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)







# -------------------------------------------------------------------
# Embeddings and Vector Store
# -------------------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"  # or "text-embedding-3-small"
)



# -------------------------------------------------------------------
# Google Token Verification
# -------------------------------------------------------------------


def verify_google_token(token):
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            GOOGLE_CLIENT_ID
        )

        return {
            "sub": idinfo["sub"],              # <-- REQUIRED
            "user_id": idinfo["sub"],          # optional alias
            "email": idinfo["email"],
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
        }

    except ValueError:
        print("Invalid Google token:", token)
        return None


# -------------------------------------------------------------------
# JWT Creation
# -------------------------------------------------------------------

def create_jwt(user: dict):
    print("Creating JWT for user:", user)
    signing_key = SECRET_KEY + user["jwt_secret"]  # <-- IMPORTANT

    payload = {
        "user_id": user["_id"],
        "email": user["email"],
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "iat": datetime.now(timezone.utc)
    }

    return jwt.encode(payload, signing_key, algorithm="HS256")


# -------------------------------------------------------------------
# User Management
# -------------------------------------------------------------------


def get_or_create_user(idinfo):
    user_id = idinfo["sub"]

    existing_user = users_collection.find_one({"_id": user_id})

    if existing_user:
        return existing_user

    new_user = {
        "_id": user_id,
        "email": idinfo["email"],
        "name": idinfo.get("name"),
        "picture": idinfo.get("picture"),
        "messages": [{"role": "AI", "content": f"Hello {idinfo.get('name')}! How can I assist you with your documents today?"}],
        "jwt_secret": token_hex(32),
        "document": {"name": "","chunks": []}
    }

    users_collection.insert_one(new_user)   
    

    return new_user



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
# Message Management
# -------------------------------------------------------------------


def add_message_to_user(user, role: str, content: str):
    if user is None:
        return

    users_collection.update_one(
        {"_id": user["_id"]},
        {"$push": {"messages": {"role": role, "content": content}}}
    )


# -------------------------------------------------------------------
# PDF Query & Processing
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

def query(question: str, token: str = None):
    user = verify_jwt(token)

    # Ensure user has uploaded a document
    user_doc = user.get("document")
    if not user_doc or "chunks" not in user_doc:
        raise HTTPException(400, "You must upload a document first.")

    chunks = user_doc["chunks"]      # list of text chunks
    doc_name = user_doc["name"]

    # Build FAISS vector store IN MEMORY
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Retrieval: get most relevant chunks
    docs = vector_store.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer ONLY using the context."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ]
    )

    ai_text = response.choices[0].message.content

    # Save chat history
    add_message_to_user(user, "user", question)
    add_message_to_user(user, "AI", ai_text)

    return {
        "answer": ai_text,
        "document_used": doc_name,
        "matched_chunks": len(docs),
    }

# -------------------------------------------------------------------
# Document Processing
# -------------------------------------------------------------------

def process_document(path: str) -> List[str]:
    loader = PyPDFLoader(path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    return [c.page_content for c in chunks]


# -------------------------------------------------------------------
# Upload Endpoint   
# -------------------------------------------------------------------


@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    user = verify_jwt(authorization.replace("Bearer ", ""))

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed.")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_document(path)

    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "document": {
                    "name": file.filename,
                    "chunks": chunks,
                }
            }
        }
    )

    return {"message": "Document saved successfully."}

@app.post("/removefile/")
async def delete_document(
    authorization: str = Header(None)
):
    user = verify_jwt(authorization.replace("Bearer ", ""))

    

    users_collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "document": {
                    "name": "",
                    "chunks": [],
                }
            }
        }
    )

    return {"message": "Document deleted successfully."}

# -------------------------------------------------------------------
# Query Endpoint
# -------------------------------------------------------------------


@app.post("/query/")
async def ask_question(
    request: QueryRequest, 
    authorization: str = Header(None)
):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    token = authorization.replace("Bearer ", "")

    verify_jwt(token)

    return query(request.question, token)


# -------------------------------------------------------------------
# User Auth Endpoint
# -------------------------------------------------------------------


@app.post("/userauth/")
async def user_auth(user: dict):
    # Implement user authentication logic here
    idinfo = verify_google_token(user.get("token"))


    print ("User auth with idinfo:", idinfo)

    if not idinfo:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    
    user_record = get_or_create_user(idinfo)
    jwt_token = create_jwt(user_record)
    return {"token": jwt_token, "user": user_record}
        
# -------------------------------------------------------------------
# Signout Endpoint
# -------------------------------------------------------------------

@app.post("/signout/")
async def user_signout(token: str):
    user = verify_jwt(token)

    new_secret = token_hex(32)

    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"jwt_secret": new_secret}}
    )

    return {"message": "Signed out successfully"}

# -------------------------------------------------------------------
# Me Endpoint
# -------------------------------------------------------------------

@app.get("/me")
async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    token = authorization.replace("Bearer ", "")
    user = verify_jwt(token)

    # refetch from DB to guarantee it's up-to-date
    fresh = users_collection.find_one({"_id": user["_id"]})

    return fresh
        
# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
