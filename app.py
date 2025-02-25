
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document  # Import for handling DOCX files
import os
import string
import random

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

origins = ["https://docchatv1.onrender.com"]  # Allowed origins (Add more if needed)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
PINE_CONE_KEY = os.getenv("PINE_CONE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINE_CONE_KEY or not GROQ_API_KEY:
    raise Exception("API keys not set.")

# Initialize Pinecone and SentenceTransformer
pc = Pinecone(api_key=PINE_CONE_KEY)
index = pc.Index('example-index')
model = SentenceTransformer('all-mpnet-base-v2')

# Global in-memory store for conversation memory
memories = {}

def randomIdGenerate():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    return text_splitter.split_text(text)

def embedTheText(text_chunks):
    embeddings = model.encode(text_chunks)
    metadata_list = [{"text": chunk} for chunk in text_chunks]
    ids = [f'id-{randomIdGenerate()}' for _ in text_chunks]
    vectors = [
        {
            'id': id_,
            'values': embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            'metadata': metadata
        }
        for id_, embedding, metadata in zip(ids, embeddings, metadata_list)
    ]
    return vectors

def saveInPinecone(vectors, namespace):
    index.upsert(vectors=vectors, namespace=namespace)

def extract_text_from_docx(docx_file: bytes) -> str:
    """Extracts text from a DOCX file."""
    doc = Document(BytesIO(docx_file))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    text = ""
    
    for file in files:
        file_bytes = await file.read()
        if file.filename.endswith(".pdf"):
            
            pdf_reader = PdfReader(BytesIO(file_bytes))
            for page in pdf_reader.pages:
                
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file.filename.endswith(".docx"):
            
            text += extract_text_from_docx(file_bytes) + "\n"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from files.")

    chunks = get_text_chunks(text)
    vectors = embedTheText(chunks)
    namespace = randomIdGenerate()
    saveInPinecone(vectors, namespace)

    memories[namespace] = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True
    )

    return JSONResponse(content={"namespace": namespace, "message": "Files processed successfully"})

@app.post("/chat")
async def chat(user_question: str = Form(...), namespace: str = Form(...)):
    if namespace not in memories:
        memories[namespace] = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", return_messages=True
        )

    query_embedding = model.encode([user_question])[0].tolist()
    result = index.query(
        top_k=5, namespace=namespace, vector=query_embedding,
        include_values=True, include_metadata=True
    )

    matched_info = ' '.join(item['metadata']['text'] for item in result['matches'])
    context = f"Information: {matched_info}"

    sys_prompt = f"""
    Instructions:
    - Utilize the provided context only.
    - Do not answer unrelated questions.
    - Format responses properly with headings and spacing.
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=sys_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])

    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="qwen-2.5-coder-32b")
    conversation = LLMChain(
        llm=groq_chat, prompt=prompt, verbose=False, memory=memories[namespace]
    )

    response = conversation.predict(human_input=user_question)
    return JSONResponse(content={"question": user_question, "answer": response})

if __name__ == "__main__":    
    app.run(debug=True)
