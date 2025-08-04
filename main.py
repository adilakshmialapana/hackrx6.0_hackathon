import os
import io
import json
import requests
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, status, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# RAG-specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Set Gemini API key directly
api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

# --- FastAPI App Instance ---
app = FastAPI(
    title="HackRX 6.0 - Intelligent Query-Retrieval System",
    description="Solution optimized for speed and accuracy using Gemini 2.5 Flash.",
    version="1.0.0"
)

# Security scheme
security = HTTPBearer()

# --- Pydantic Data Models (Exact Input & Output) ---
class DocumentInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answers(BaseModel):
    answers: List[str]

# --- RAG Components (in-memory for the hackathon) ---
vector_store = None
llm = None
retriever = None

# --- Core RAG Initialization Function ---
def initialize_rag_system(document_url: str):
    """Initializes the RAG system by downloading, parsing, and indexing a document."""
    global vector_store, llm, retriever
    
    try:
        # Download PDF from URL
        response = requests.get(str(document_url))
        response.raise_for_status()
        
        # Save PDF to temporary file
        pdf_content = io.BytesIO(response.content)
        with open("temp_document.pdf", "wb") as f:
            f.write(pdf_content.getvalue())
        
        # Load PDF using PyPDF2 (simpler approach for Windows compatibility)
        import PyPDF2
        documents = []
        try:
            with open("temp_document.pdf", "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        from langchain.schema import Document
                        documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
        except Exception as pdf_error:
            print(f"Error loading PDF: {pdf_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load PDF: {str(pdf_error)}"
            )
        
        # Split documents into chunks with better parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini API key not found. Please check your configuration."
            )
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Initialize LLM with Gemini 2.5 Flash model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Changed to the 2.5 Flash model
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        
        # Set up retriever with balanced search parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6 # Get more chunks for better coverage
            }
        )
        
        # Clean up temporary file
        if os.path.exists("temp_document.pdf"):
            os.remove("temp_document.pdf")
            
        print(f"RAG system initialized with document from: {document_url}")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize RAG system: {str(e)}"
        )

# --- Simple Answer Generation Function with Semaphore ---
async def get_answer_for_question(question: str, semaphore: asyncio.Semaphore):
    """Get answer for a single question using RAG, with a semaphore to control concurrency."""
    async with semaphore:
        try:
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(question)
            
            # If no relevant docs found, try with a more general search
            if not docs:
                fallback_retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}
                )
                docs = fallback_retriever.get_relevant_documents(question)
            
            # Create context from retrieved documents
            if docs:
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    context_parts.append(f"Document Chunk {i}:\n{doc.page_content}")
                context = "\n\n".join(context_parts)
            else:
                context = "No relevant information found in the document."
            
            # Construct prompt for accurate answers
            prompt = f"""You are an expert system designed to provide accurate answers based on the provided context.

IMPORTANT RULES:
1. Answer using information present in the context
2. If the context contains the answer, provide it clearly and concisely
3. If the context does not contain the answer, respond with "The provided context does not contain the answer to this question."
4. Do not add external knowledge or assumptions
5. Keep answers focused and relevant to the question

Context:
{context}

Question: {question}

Answer:"""
            
            # Get answer from LLM
            response = await llm.ainvoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
            return f"Error processing question: {str(e)}"


@app.post("/api/v1/hackrx/run", response_model=Answers, status_code=status.HTTP_200_OK)

async def run_submissions(data: DocumentInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Processes a document and answers a list of questions using a concurrent RAG system.
    """
    # Authentication check
    if credentials.credentials != "f072e58e6d9a51de69f3f1d1a0e267f663a545d4c3b4edda40dba2e631f1ee73":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    
    try:
        if data.documents:
            initialize_rag_system(data.documents)
        
        if not retriever:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG system not initialized. A document URL is required."
            )

        # Create a semaphore with a value of 5 for more aggressive concurrency
        # with the Gemini 2.5 Flash model. You may adjust this based on your
        # specific API quota.
        semaphore = asyncio.Semaphore(5) 
        
        # Create a list of tasks for all questions
        tasks = [get_answer_for_question(question, semaphore) for question in data.questions]
        
        # Run all tasks concurrently
        all_answers = await asyncio.gather(*tasks)
        
        return Answers(answers=all_answers)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in processing: {e}")
        print(f"Full traceback: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in processing: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
