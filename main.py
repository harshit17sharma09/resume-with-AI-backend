from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import logging
import os
import tempfile
import hashlib
import re
import pdfplumber
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from google.cloud import firestore
from firebase_admin import credentials, initialize_app, firestore as admin_firestore
import firebase_admin
from utils.jwt_handler import decode_access_token, create_access_token

# Initialize Firebase Admin with explicit project ID
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv('FIREBASE_ADMIN_CREDENTIALS'))
    firebase_app = initialize_app(cred, {
        'projectId': 'resume-chatbot-25207'
    })

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Firestore client using Firebase Admin
db = admin_firestore.client()

# Initialize LangChain components
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.7)

# Initialize storage dictionaries
vector_stores = {}
conversation_memories = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Models
class UserModel(BaseModel):
    email: str

class AccessRequestModel(BaseModel):
    email: str

class ChatMessage(BaseModel):
    role: str
    content: str

# Storage dictionaries
chat_histories: dict[str, list[ChatMessage]] = {}
active_resumes: dict[str, str] = {}  # email -> resume_content_hash

# Middleware: Validate JWT token
def validate_token(token: str = Header(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token.split('Bearer ')[-1].strip()
        return decode_access_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


# Login Endpoint
@app.post("/login")
def login(user: UserModel):
    """Simulate Google Sign-In and issue a JWT token."""
    token = create_access_token({"sub": user.email}, timedelta(minutes=30))
    return {"access_token": token, "token_type": "bearer"}

# Upload Resume Endpoint
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), authorization: str = Header(None, alias="Authorization")):
    """Handle resume uploads."""
    # Extract and validate the token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")
    token = authorization.split("Bearer ")[-1].strip()
    payload = decode_access_token(token)

    # Use payload data (e.g., email)
    email = payload.get("sub")
    content = await file.read()
    return {"message": f"Resume uploaded for {email}", "filename": file.filename}


# Request Access Endpoint
@app.post("/request-access")
async def request_access(authorization: str = Header(None, alias="Authorization")):
    """
    Log an access request in Firestore.
    """
    try:
        # Validate token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")

        token = authorization.split("Bearer ")[-1].strip()
        payload = decode_access_token(token)
        email = payload.get("sub")

        if not email:
            raise HTTPException(status_code=401, detail="Invalid token: email missing")

        # Check if there's an existing request and it's within 1 minute
        doc_ref = db.collection("access_requests").document(email)
        doc_snap = doc_ref.get()

        if doc_snap.exists:
            data = doc_snap.to_dict()
            last_updated = data.get("lastUpdatedAt")
            
            if last_updated:
                # Convert Firestore timestamp directly
                last_updated_dt = last_updated
                current_time = datetime.now(timezone.utc)
                
                # Check if last request was within 1 minute
                time_diff = current_time - last_updated_dt
                if time_diff.total_seconds() < 60:  # 1 minute = 60 seconds
                    seconds_left = int(60 - time_diff.total_seconds())
                    raise HTTPException(
                        status_code=429,
                        detail=f"Please wait {seconds_left} seconds before requesting access again"
                    )

        # Save to Firestore with current timestamp
        doc_ref.set({
            "email": email,
            "hasAccess": False,
            "lastUpdatedAt": firestore.SERVER_TIMESTAMP,
            "createdAt": firestore.SERVER_TIMESTAMP if not doc_snap.exists else doc_snap.get("createdAt")
        }, merge=True)

        return {"message": "Access request raised", "email": email}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in request_access: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))


# Check Access Endpoint
@app.get("/check-access")
async def check_access(authorization: str = Header(None, alias="Authorization")):
    """
    Check if the user has access and return request status.
    """
    try:
        # Validate the token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization token required")

        token = authorization.split("Bearer ")[-1].strip()
        payload = decode_access_token(token)
        email = payload.get("sub")

        if not email:
            raise HTTPException(status_code=401, detail="Invalid token: email missing")

        # Retrieve access status from Firestore
        doc_ref = db.collection("access_requests").document(email)
        doc_snap = doc_ref.get()

        if not doc_snap.exists:
            return {
                "hasAccess": False,
                "canRequest": True,
                "lastUpdatedAt": None,
                "minutesUntilNextRequest": 0
            }

        access_data = doc_snap.to_dict()
        last_updated = access_data.get("lastUpdatedAt")
        
        # Calculate time until next request
        can_request = True
        minutes_until_next = 0
        
        if last_updated:
            current_time = datetime.now(timezone.utc)
            time_diff = current_time - last_updated
            if time_diff.total_seconds() < 60:  # 1 minute
                can_request = False
                seconds_until_next = int(60 - time_diff.total_seconds())
                minutes_until_next = max(1, int(seconds_until_next / 60))

        return {
            "hasAccess": access_data.get("hasAccess", False),
            "canRequest": can_request,
            "lastUpdatedAt": last_updated.isoformat() if last_updated else None,
            "minutesUntilNextRequest": minutes_until_next
        }

    except Exception as e:
        logger.error(f"Error in check_access: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))


# Chatbot Endpoint
# Chat API endpoint
@app.post("/chat")
async def chat_with_resume(
    query: str = Form(...),
    file: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization")
):
    try:
        logger.info("Chat request received")
        logger.debug(f"Authorization header: {authorization[:20]}...")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        # Verify file size
        file_size = 0
        chunk_size = 1024
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="File too large")
        
        await file.seek(0)
        
        # Refresh token if needed
        valid_token = await refresh_token_if_needed(authorization)
        logger.info("Token validated/refreshed successfully")
        
        # Get email from the valid token
        payload = decode_access_token(valid_token)
        email = payload.get("sub")
        
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token: email missing")
        
        logger.info(f"Processing chat for user: {email}")
        
        # Process the resume
        resume_content = await process_resume(file, email)
        
        # Create a chat prompt that includes the resume content and query
        chat_prompt = f"""
        You are an AI assistant analyzing a resume. The resume content is available to you, but first engage naturally with the user's query.

        Resume content for reference:
        {resume_content}

        User's question: {query}

        Guidelines for responding:
        1. If the user is making general conversation (like saying hello), respond naturally without jumping into resume analysis
        2. If the user asks about the resume or for specific advice, provide detailed analysis and suggestions
        3. Only provide resume-specific feedback when explicitly asked
        4. Keep the tone friendly and conversational
        """

        # Get response from OpenAI
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo"
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a friendly AI assistant who can help with resume analysis. Respond naturally to conversation while providing resume advice only when specifically asked."
                    },
                    {"role": "user", "content": chat_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            response_text = chat_completion.choices[0].message.content
            logger.info("Successfully generated chat response")
            
        except Exception as e:
            logger.error(f"Error getting chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating response")
        
        return {
            "response": response_text,
            "new_token": valid_token if valid_token != authorization.split("Bearer ")[-1].strip() else None
        }

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_resume(file: UploadFile, email: str):
    """Process the uploaded resume file."""
    try:
        logger.info(f"Processing resume for {email}")
        
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            
            # Read the content from the UploadFile
            content = await file.read()
            if not content:
                logger.error("Uploaded file is empty")
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
            # Write content to temporary file
            temp_file.write(content)
            temp_file.flush()  # Ensure all data is written
            
            logger.info(f"Temporary file created at: {temp_path}")
            
            # Reset file pointer for future reads
            await file.seek(0)
            
            # Load and process the PDF
            try:
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                logger.info(f"Successfully loaded PDF with {len(documents)} pages")
                
                # Process the documents as needed
                text_content = " ".join([doc.page_content for doc in documents])
                
                return text_content
                
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                raise
            
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
    finally:
        # Clean up temporary file
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

def generate_response_with_context(
    query: str, 
    resume_data: dict, 
    embeddings: dict, 
    chat_history: list[ChatMessage]
):
    logger.info("Generating response with context...")
    try:
        relevant_section, similarity = find_relevant_section(query, embeddings)
        relevant_content = next(sec["content"] for sec in resume_data["sections"] if sec["title"] == relevant_section)
        
        # Create the conversation history for the prompt
        conversation_history = "\n".join([
            f"{msg.role}: {msg.content}" 
            for msg in chat_history[-6:]
        ])

        # Use OpenAI GPT to generate a response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": f"""You are an expert technical recruiter responsible for analyzing resumes.
                    Provide concise, professional responses based only on the information in the resume.
                    If information is not present in the resume, clearly state that.
                    
                    Resume Section: {relevant_content}
                    
                    Previous Conversation:
                    {conversation_history}"""
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            temperature=0.7,
            max_tokens=150
        )

        content = response.choices[0].message.content
        return content

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

# Add an endpoint to clear chat history
@app.post("/clear-chat")
async def clear_chat_history(authorization: str = Header(None, alias="Authorization")):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization token required")

    token = authorization.split("Bearer ")[-1].strip()
    
    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token: email missing")

        if email in chat_histories:
            chat_histories[email] = []
        
        return {"message": "Chat history cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def is_valid_resume(text: str) -> bool:
    """
    Validate if the document appears to be a resume by checking for common resume sections
    and keywords.
    """
    # Common resume sections and keywords
    resume_indicators = [
        r'experience',
        r'education',
        r'skills',
        r'work\s+history',
        r'employment',
        r'qualifications',
        r'projects',
        r'achievements',
        r'certifications',
        r'summary',
        r'objective'
    ]

    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Count how many resume indicators are present
    indicator_count = sum(1 for indicator in resume_indicators 
                        if re.search(indicator, text_lower))

    # Require at least 3 resume sections to be present
    return indicator_count >= 3

async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Enhanced PDF text extraction with better formatting preservation.
    """
    try:
        # Read the entire file content at once
        content = await file.read()
        
        # Create temporary file and write content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        text_chunks = []
        
        try:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        text_chunks.append(text)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if not text_chunks:
            return ""

        # Join text with proper spacing
        full_text = '\n\n'.join(text_chunks)

        # Clean up common PDF extraction issues
        cleaned_text = re.sub(r'\s+', ' ', full_text)  # Remove excessive whitespace
        cleaned_text = re.sub(r'([.!?])\s*', r'\1\n', cleaned_text)  # Add newlines after sentences
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Clean up multiple newlines

        return cleaned_text.strip()

    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF file: {str(e)}")



def validate_token(token: str = Header(None)):
    if not token:
        print("Authorization header missing")
        raise HTTPException(status_code=401, detail="Authorization token required")
    print(f"Token received: {token}")
    return decode_access_token(token)





@app.get("/validate-token")
async def validate_token_endpoint(authorization: str = Header(None, alias="Authorization")):
    """
    Validate the JWT token provided in the Authorization header.
    """
    print(f"Authorization header received: {authorization}")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")

    # Extract the token after "Bearer "
    token = authorization.split("Bearer ")[-1].strip()
    print(f"Token extracted: {token}")

    # Validate and decode the token
    payload = decode_access_token(token)
    return {"message": "Token is valid", "payload": payload}


# Helper function to extract text from a PDF
# Function to extract text from a PDF file (for the sake of processing)
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        if len(pdf.pages) > 2:  # Limit to 2 pages
            st.error("Uploaded PDF exceeds the 2-page limit. Please upload a shorter resume.")
            return None
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()


# Helper function to parse resume sections
# Function to parse the resume text into sections using headers
def parse_resume_text_with_headers(text):
    section_headers = ["About Me", "Skills", "Experience", "Education", "Projects", "Certifications"]
    pattern = "|".join([f"\\b{header}\\b" for header in section_headers])

    sections = re.split(pattern, text, flags=re.IGNORECASE)
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    parsed_sections = []
    for i, section in enumerate(sections):
        if i == 0 and section.strip():
            parsed_sections.append({"title": "Introduction", "content": section.strip()})
        elif i < len(matches):
            parsed_sections.append({"title": matches[i].strip(), "content": section.strip()})
    
    return {"sections": parsed_sections}

# Helper function to generate embeddings
# def get_embedding(text, model="text-embedding-ada-002"):
#     try:
#         print(f"Generating embedding for: {text[:100]}...")  # Log the text being passed
#         response = client.embeddings.create(
#             model=model,
#             input=text  # Ensure input is passed here
#         )
#         print(f"Embedding response: {response}")
#         # embedding = response['data'][0]['embedding']
#         embedding = response.data[0].embedding
#         return embedding
#         # return response["data"][0]["embedding"]
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         raise HTTPException(status_code=400, detail=f"Error generating embedding: {e}")


# Helper function to find the most relevant section
def find_relevant_section(query, embeddings):
    logger.info(f"Inside find relevant section")
    logger.info(f"Embedding input text in section: {query}")
    query_embedding = get_embedding(query)
    logger.info("1")
    section_titles = list(embeddings.keys())
    logger.info("2")
    section_embeddings = np.array(list(embeddings.values()))
    logger.info("3")
    similarities = cosine_similarity([query_embedding], section_embeddings).flatten()
    logger.info("4")
    most_relevant_idx = np.argmax(similarities)
    logger.info("5")
    return section_titles[most_relevant_idx], similarities[most_relevant_idx]


def generate_section_embeddings(resume_data):
    logger.info(f"Inside generate section embeddings")
    embeddings = {}
    for section in resume_data["sections"]:
        embeddings[section["title"]] = get_embedding(section["content"])
    return embeddings


def generate_response(query, resume_data, embeddings):
    logger.info("Generating response...")
    try:
        relevant_section, similarity = find_relevant_section(query, embeddings)
        relevant_content = next(sec["content"] for sec in resume_data["sections"] if sec["title"] == relevant_section)
        
        # Use OpenAI GPT to generate a response
        response = client.chat.completions.create(
            model="gpt-4",  # Changed from gpt-4o to gpt-4
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert technical recruiter responsible for analyzing resumes.
                    Provide concise, professional responses based only on the information in the resume.
                    If information is not present in the resume, clearly state that."""
                },
                {
                    "role": "user", 
                    "content": f"Question: {query}\nRelevant Resume Section: {relevant_content}"
                }
            ],
            temperature=0.7,
            max_tokens=150
        )

        content = response.choices[0].message.content
        return content

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")


def get_embedding(text, model="text-embedding-ada-002"):
    logger.info(f"Inside get_embedding")
    logger.info(f"get_Embedding input text: {text}")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Input text for embedding is empty.")
    try:
        response = client.embeddings.create(input=text.strip(),model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=400, detail=f"Error generating embedding: {e}")

@app.post("/reset-session")
async def reset_session(authorization: str = Header(None, alias="Authorization")):
    """Reset user's session to allow uploading a new resume."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization token required")

    token = authorization.split("Bearer ")[-1].strip()
    
    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token: email missing")

        # Clear chat history
        if email in chat_histories:
            chat_histories[email] = []
        
        # Clear active resume
        if email in active_resumes:
            del active_resumes[email]

        return JSONResponse(
            status_code=200,
            content={"message": "Session reset successfully"}
        )

    except Exception as e:
        logger.error(f"Reset session error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_resume_hash(content: str) -> str:
    """Generate a unique hash for the resume content."""
    return hashlib.sha256(content.encode()).hexdigest()

def validate_and_get_email(authorization: str) -> str:
    """Helper function to validate token and return email"""
    if not authorization or not authorization.startswith("Bearer "):
        logger.error("Invalid or missing Authorization header")
        raise HTTPException(status_code=401, detail="Authorization token required")

    try:
        token = authorization.split("Bearer ")[-1].strip()
        logger.info("Validating token...")
        
        payload = decode_access_token(token)
        email = payload.get("sub")
        
        if not email:
            logger.error("No email found in token payload")
            raise HTTPException(status_code=401, detail="Invalid token: email missing")
        
        logger.info(f"Token validated for email: {email}")
        return email
        
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))

# Add new helper function
async def refresh_token_if_needed(authorization: str) -> str:
    """
    Check token validity and refresh if expired but user has access.
    Returns the valid token (either existing or new).
    """
    try:
        if not authorization or not authorization.startswith("Bearer "):
            logger.error(f"Invalid authorization header: {authorization}")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        token = authorization.split("Bearer ")[-1].strip()
        logger.info(f"Token to verify: {token[:10]}...") # Log first 10 chars of token
        
        if not token or len(token.split('.')) != 3:  # JWT tokens should have 3 segments
            logger.error(f"Invalid token format. Token segments: {len(token.split('.'))}")
            raise HTTPException(status_code=401, detail="Invalid token format")

        try:
            # Try to decode the existing token with expiration check
            payload = decode_access_token(token, verify_expiration=True)
            logger.info("Token is valid")
            return token  # Token is still valid
        except HTTPException as token_error:
            if "expired" in str(token_error.detail).lower():
                # Token is expired, check if user has access
                logger.info("Token expired, checking user access")
                
                try:
                    # Get email from expired token without expiration verification
                    expired_payload = decode_access_token(token, verify_expiration=False)
                    email = expired_payload.get("sub")
                    
                    if not email:
                        logger.error("No email found in token payload")
                        raise HTTPException(status_code=401, detail="Invalid token format")

                    logger.info(f"Checking access for email: {email}")
                    # Check user access in Firestore
                    doc_ref = db.collection("access_requests").document(email)
                    doc_snap = doc_ref.get()
                    
                    if doc_snap.exists and doc_snap.to_dict().get("hasAccess", False):
                        # User has access, create new token
                        logger.info(f"Refreshing token for user: {email}")
                        new_token = create_access_token(
                            {"sub": email},
                            timedelta(hours=24)
                        )
                        return new_token
                    else:
                        logger.error(f"Access denied for email: {email}")
                        raise HTTPException(status_code=401, detail="Access revoked or not granted")
                except Exception as inner_e:
                    logger.error(f"Error processing expired token: {str(inner_e)}")
                    raise HTTPException(status_code=401, detail="Error processing token")
            else:
                logger.error(f"Token error: {str(token_error.detail)}")
                raise token_error
                
    except Exception as e:
        logger.error(f"Error in refresh_token_if_needed: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))