from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnableLambda
from langchain_core.messages import BaseMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define model and embedding
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant like Jarvis in Tony Stark’s lab. Think step-by-step using your own knowledge. Address the user as 'Sir'. Here is the context:\n\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain
stuff_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Initialize FastAPI
app = FastAPI(
    title="LangChain PDF QA API",
    version="1.0",
    description="Upload a PDF and ask questions about its content"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)# Global chat history store
chat_history = {}

# Helper to get chat history per session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]

def serialize_chat_messages(messages: list[BaseMessage]) -> list[dict]:
    return [{"type": msg.__class__.__name__, "content": msg.content} for msg in messages]

@app.post("/upload_pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_input: str = Form(...)
):
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and split PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        chunks = splitter.split_documents(docs)

        # Create vector DB and retriever
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever()

        # Setup retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, stuff_doc_chain)
        chain_with_postprocess = retrieval_chain | RunnableLambda(
            lambda x: {"output": x["answer"], "context_data": x}
        )

        # Setup history-enabled chain
        global history_chain
        history_chain = RunnableWithMessageHistory(
            chain_with_postprocess,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        # Invoke the chain
        result = await history_chain.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        context_docs = result.get("context_data", {}).get("context", [])
        serialized_context = [doc.page_content for doc in context_docs]
        history_obj = chat_history.get(session_id, None)
        serialized_history = (
            serialize_chat_messages(history_obj.messages)
            if history_obj else []
        )
        return JSONResponse(content={
            "answer": result.get("output"),
            "context": serialized_context,
            "history": serialized_history
        })
        # return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.post("/invoke_query/")
async def upload_pdf(
    session_id: str = Form(...),
    user_input: str = Form(...)
):

        # Invoke the chain
        if history_chain == None:
            return JSONResponse(content={"error": "No history chain found"}, status_code=500)
        result = await history_chain.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        context_docs = result.get("context_data", {}).get("context", [])
        serialized_context = [doc.page_content for doc in context_docs]
        history_obj = chat_history.get(session_id, None)
        serialized_history = (
            serialize_chat_messages(history_obj.messages)
            if history_obj else []
        )
        return JSONResponse(content={
            "answer": result.get("output"),
            "context": serialized_context,
            "history": serialized_history
        })