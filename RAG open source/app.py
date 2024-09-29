from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, List
from datetime import datetime
import textwrap
import numpy as np
import PyPDF2
import io

# Initialize FastAPI
app = FastAPI()

# Hyperparameters
CHUNK_SIZE = 500  # Hyperparameter for chunking documents
TOP_K = 3         # Number of top similar items to retrieve
MAX_TOKENS = 2048  # Model's maximum context window

# Load Llama model with extended context window
llm = Llama.from_pretrained(
    repo_id="rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia",
    filename="unsloth.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=MAX_TOKENS
)

# Load text embedding model
text_embedder = Llama.from_pretrained(
    repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    embedding=True
)

# Database credentials
DB_NAME = "rag_open_source"
DB_USER = "your_user"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "8012"

# Database connection string
DATABASE_URL = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

# Define request and response models
class ChatRequest(BaseModel):
    instruction: str
    input_data: str = ""

class ChatResponse(BaseModel):
    response: str
    chat_history_id: int

class ReactionRequest(BaseModel):
    chat_history_id: int
    reaction: str  # "like" or "dislike"

class RegenerateRequest(BaseModel):
    chat_history_id: int

# Function to check cache
def get_cached_response(instruction: str, input_data: str):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        query = """
        SELECT ch.response, ch.id FROM chat_history ch
        JOIN cached_chat cc ON cc.chat_history_id = ch.id
        WHERE ch.instruction = %s AND ch.input_data = %s
        """
        cursor.execute(query, (instruction, input_data))
        result = cursor.fetchone()
        conn.close()
        return result  # result will have 'response' and 'id'
    except Exception as e:
        print(f"Error accessing database: {e}")
        return None

# Function to store response in cache
def cache_response(chat_history_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        query = "INSERT INTO cached_chat (chat_history_id) VALUES (%s)"
        cursor.execute(query, (chat_history_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error inserting into database: {e}")

# Function to save chat history with vectors
def save_chat_history(instruction: str, input_data: str, response: str, input_vector, response_vector):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        query = """
        INSERT INTO chat_history (instruction, input_data, response, input_vector, response_vector)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """
        cursor.execute(query, (instruction, input_data, response, input_vector.tolist(), response_vector.tolist()))
        chat_history_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return chat_history_id
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return None

# Function to add reaction (like, dislike, regenerate)
def add_reaction(chat_history_id: int, reaction: str):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        query = "INSERT INTO analytics (type, chat_history_id) VALUES (%s, %s)"
        cursor.execute(query, (reaction, chat_history_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving reaction: {e}")

# Function to get chat history by ID
def get_chat_history_by_id(chat_history_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        query = "SELECT instruction, input_data, response FROM chat_history WHERE id = %s"
        cursor.execute(query, (chat_history_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return None

# Function to get all reactions based on reaction_type and optional datetime filters
def get_all_reactions(reaction_type: str, start_datetime: Optional[str], end_datetime: Optional[str]):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        query = "SELECT * FROM analytics WHERE type = %s"
        params = [reaction_type]
        
        # Filter by start and end datetime
        if start_datetime:
            query += " AND created_at >= %s"
            params.append(start_datetime)
        
        if end_datetime:
            query += " AND created_at <= %s"
            params.append(end_datetime)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Error retrieving reactions: {e}")
        return None

# Function to embed text using the text_embedder model
def embed_text(text: str):
    embeddings = text_embedder.embed(text)
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings

# Function to search knowledge base using embeddings
def search_knowledge_base(query_embedding, top_k=TOP_K):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        query = """
        SELECT chunk_text FROM knowledge_vector
        ORDER BY chunk_vector <-> (%s)::vector LIMIT %s;
        """
        cursor.execute(query, (query_embedding.tolist(), top_k))
        results = cursor.fetchall()
        conn.close()
        return [row['chunk_text'] for row in results]
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []

# Function to search chat history using embeddings
def search_chat_history(query_embedding, top_k=TOP_K):
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        query = """
        SELECT instruction, input_data, response FROM chat_history
        WHERE input_vector IS NOT NULL
        ORDER BY input_vector <-> (%s)::vector LIMIT %s;
        """
        cursor.execute(query, (query_embedding.tolist(), top_k))
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Error searching chat history: {e}")
        return []

# Function to count tokens in text
def count_tokens(text: str):
    # Placeholder function; implement actual token counting based on your tokenizer
    # For example, use GPT2Tokenizer or a similar tokenizer compatible with your model
    return len(text.split())

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    # Check if response is already cached
    cached = get_cached_response(request.instruction, request.input_data)
    if cached:
        return ChatResponse(response=cached['response'], chat_history_id=cached['id'])
    
    # Embed the input instruction and input data
    combined_input = request.instruction + " " + request.input_data
    input_embedding = embed_text(combined_input)

    # Search knowledge base for relevant chunks
    relevant_chunks = search_knowledge_base(input_embedding)

    # Search chat history for relevant past chats
    # relevant_chats = search_chat_history(input_embedding)
    chat_history_context = ""
    # for chat in relevant_chats:
    #     chat_history_context += f"Instruction: {chat['instruction']}\nInput: {chat['input_data']}\nResponse: {chat['response']}\n\n"

    # Define preprompt
    preprompt = "Kamu adalah chatbot bernama KulCi, KulCi merupakan customer service yang melayani tentang Kuliner Di Cirebon. tolong hanya jawab sesuai konteks yang diberikan"

    # Construct system content
    knowledge_context = "\n".join(relevant_chunks)
    system_content = preprompt + "\n\n" + knowledge_context

    # User input
    user_input = request.instruction + " " + request.input_data

    # Construct messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]


    print(messages)

    # Generate the chat completion
    result = llm.create_chat_completion(
        messages=messages,
        stop = ["[/INST]"] # Stopping generate answer
    )

    print(result)

    # Extract and return the result
    response_text = result['choices'][0]['message']['content'].replace('<<SYS>>','').strip()

    # Embed the response
    response_embedding = embed_text(response_text)

    # Save the chat history and get chat_history_id
    chat_history_id = save_chat_history(request.instruction, request.input_data, response_text, input_embedding, response_embedding)

    # Cache the response
    cache_response(chat_history_id)

    return ChatResponse(response=response_text, chat_history_id=chat_history_id)

@app.post("/regenerate", response_model=ChatResponse)
async def regenerate_chat(request: RegenerateRequest):
    # Retrieve the original chat history
    history = get_chat_history_by_id(request.chat_history_id)
    if not history:
        raise HTTPException(status_code=404, detail="Chat history not found")

    # Embed the input instruction and input data
    combined_input = history['instruction'] + " " + history['input_data']
    input_embedding = embed_text(combined_input)

    # Search knowledge base for relevant chunks
    relevant_chunks = search_knowledge_base(input_embedding)

    # Search chat history for relevant past chats
    # relevant_chats = search_chat_history(input_embedding)
    chat_history_context = ""
    # for chat in relevant_chats:
    #     chat_history_context += f"Instruction: {chat['instruction']}\nInput: {chat['input_data']}\nResponse: {chat['response']}\n\n"

    # Define preprompt
    preprompt = "Kamu adalah seorang chatbot yang ditugaskan untuk menjadi customer service sebuah perusahaan bernama emerald."

    # Construct system content
    knowledge_context = "\n".join(relevant_chunks)
    system_content = preprompt + "\n\n" + knowledge_context + "\n\n" + chat_history_context

    # User input
    user_input = history['instruction'] + " " + history['input_data']

    # Construct messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]

    print(messages)

    # Count tokens and adjust if necessar

    # Generate the new chat completion
    result = llm.create_chat_completion(
        messages=messages,
         stop = ["[/INST]"]
    )

    # Extract and return the regenerated result
    response_text = result['choices'][0]['message']['content'].replace('<<SYS>>','').strip()

    # Embed the response
    response_embedding = embed_text(response_text)

    # Save the regenerated chat history and return new chat_history_id
    new_chat_history_id = save_chat_history(history['instruction'], history['input_data'], response_text, input_embedding, response_embedding)
    
    # Log the regenerate action in the analytics table with the original chat_history_id
    add_reaction(request.chat_history_id, "regenerate")

    return ChatResponse(response=response_text, chat_history_id=new_chat_history_id)

@app.post("/react")
async def react_to_chat(request: ReactionRequest):
    if request.reaction not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="Invalid reaction. Use 'like' or 'dislike'.")

    # Add reaction to the chat history
    add_reaction(request.chat_history_id, request.reaction)

    return {"message": "Reaction saved successfully."}

@app.get("/all-reactions")
async def get_all_reaction(
    reaction_type: str = Query(..., description="Filter by reaction type: like, dislike, regenerate"),
    start_datetime: Optional[str] = Query(None, description="Start datetime in the format YYYY-MM-DD HH:MM:SS"),
    end_datetime: Optional[str] = Query(None, description="End datetime in the format YYYY-MM-DD HH:MM:SS")
):
    # Get all reactions with the filters
    reactions = get_all_reactions(reaction_type, start_datetime, end_datetime)
    
    if reactions is None:
        raise HTTPException(status_code=500, detail="Error retrieving reactions.")
    
    return {"reactions": reactions}

@app.post("/add_knowledge")
async def add_knowledge(document_id: str = Query(...), file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    # Read the file content input
    try:
        if file.content_type == "text/plain":
            content = await file.read()
            document_text = content.decode('utf-8')
        elif file.content_type == "application/pdf":
            content = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            document_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    document_text += text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Split the document into chunks
    chunks = textwrap.wrap(document_text, width=CHUNK_SIZE)

    # Embed each chunk and store in the database
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Insert into knowledge table
        knowledge_query = """
        INSERT INTO knowledge (document_id)
        VALUES (%s) RETURNING id
        """
        cursor.execute(knowledge_query, (document_id,))
        knowledge_id = cursor.fetchone()[0]

        # Insert chunks into knowledge_vector table
        for chunk in chunks:
            chunk_embedding = embed_text(chunk)
            chunk_query = """
            INSERT INTO knowledge_vector (knowledge_id, chunk_text, chunk_vector)
            VALUES (%s, %s, %s)
            """
            cursor.execute(chunk_query, (knowledge_id, chunk, chunk_embedding.tolist()))

        conn.commit()
        conn.close()
        return {"message": "Knowledge added successfully."}
    except Exception as e:
        print(f"Error adding knowledge: {e}")
        raise HTTPException(status_code=500, detail="Error adding knowledge.")

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)