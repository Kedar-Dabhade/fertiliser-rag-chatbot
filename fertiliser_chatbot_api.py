from flask import Flask, request, jsonify, Response, stream_with_context, session
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from langchain.callbacks.base import BaseCallbackHandler
import queue
import threading
import re
import tiktoken  # Add this import at the top
import uuid

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
index_name = "all-products"  # or your actual index name

# Initialize Pinecone client and get index
pc = Pinecone(api_key=pinecone_api_key)
if not pc.list_indexes().names().__contains__(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,  # or your actual dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
index = pc.Index(index_name)

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

# Set up LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")

# Set up memory for chat history, specify output_key
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    max_token_limit=1250  
)

# Custom prompt for the agent
custom_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are an expert agriculture consultant. Your task is to recommend appropriate fertiliser products and answer user queries.

IMPORTANT: Do NOT repeat or rephrase the user's question in your answer. Start your answer directly with the information or advice.
Do NOT ask to consult with a local agronomist.
If the answer has more than 3-4 points summarise each of them or try fitting the answer in 2-3 points

- Identify whether the user's input is a query related to fertiliser or a general input(greetings,etc).
- If the user's input is a general input, answer it in a friendly and engaging manner.
- Even if the user's input is a general input, make sure to answer it in a way that is relevant to the fertiliser context in a respectful way.Do not pinpoint to user that it is a general input.
- Make sure to answer the question in the same language as the user's input.
- Use ONLY the following context and the chat history to answer the user's question.
- Never refer to the context as \"the context\" or \"the information\" or \"the data\" while answering the question.It will be sure that user you have enough context to answer users query upto a certain extent if not whole.Answer confidently but informatively.
- Do not recommend products that are not in the context.
- By default, provide concise, clear answers using bullet points or numbered lists for readability. If the retrieved context is big, provide a summary of the context without missing any important information and then answer the question.
- Only provide detailed explanations if the user specifically asks for more detail (e.g., 'explain in detail', 'more info', etc).
- Use line breaks and formatting to make your answer easy to read.
- If the answer is long, summarise and present it meaningfully within 5 lines, unless the user requests more detail.
- If asked about a specific product,summarise the context recieved and present it in a concise shorter format which looks good and readable.


Context:
{context}

Chat history:
{chat_history}

Question: {question}
Answer:
"""
)

# Set up Conversational Retrieval Chain (RAG with memory)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

class QueueStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.q = queue.Queue()
    def on_llm_new_token(self, token, **kwargs):
        self.q.put(token)
    def get_stream(self):
        while True:
            token = self.q.get()
            if token is None:
                break
            yield token

app = Flask(__name__)
app.secret_key = "dev-secret-key-please-change"  # Set a strong, unique key in production
CORS(app, supports_credentials=True)

@app.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

# Directory to store per-session chat histories
CHAT_HISTORY_DIR = 'sessions'
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def get_session_history():
    session_id = session['session_id']
    history_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(history_path)

@app.route('/session-chat', methods=['POST'])
def session_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    history = get_session_history()
    # Create a ConversationSummaryMemory for this session
    session_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=1250,
        chat_memory=history
    )
    # Create a ConversationalRetrievalChain for this session
    session_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=session_memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    result = session_chain.invoke({"question": user_message})
    bot_response = result["answer"]
    # Store user and bot messages in the session file
    history.add_user_message(user_message)
    history.add_ai_message(bot_response)
    return jsonify({"response": bot_response})

@app.route('/history', methods=['GET'])
def session_history():
    history = get_session_history()
    # Return as list of dicts: [{role: 'user'/'ai', content: ...}]
    messages = [
        {"role": m.type, "content": m.content} for m in history.messages
    ]
    return jsonify(messages)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    # Print the current chat history and context for debugging
    print("\n--- DEBUG: Current Chat History ---")
    print(memory.buffer)
    print("--- END DEBUG ---\n")
    # Build the full prompt as it will be sent to the LLM
    # Use the same template as custom_prompt
    context = "[context will be filled by chain]"
    chat_history = memory.buffer
    question = user_message
    prompt_text = custom_prompt.format(context=context, chat_history=chat_history, question=question)
    # Count tokens using tiktoken (make sure tiktoken is installed)
    enc = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(enc.encode(prompt_text))
    print(f"--- DEBUG: Total tokens sent to LLM: {num_tokens} ---\n")
    result = qa_chain.invoke({"question": user_message})
    return jsonify({
        "answer": result["answer"],
        # Optionally include sources:
        # "sources": [doc.metadata for doc in result["source_documents"]]
    })

@app.route('/chat-stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    handler = QueueStreamHandler()
    streaming_llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", streaming=True, callbacks=[handler])
    qa_chain_stream = ConversationalRetrievalChain.from_llm(
        llm=streaming_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

    def strip_leading_question_sentence(answer):
        # Find the first sentence (up to . ! or ?)
        match = re.match(r'^(.*?[.?!])\s*(.*)', answer.lstrip(), re.DOTALL)
        if match:
            first_sentence, rest = match.groups()
            # If the first sentence ends with a question mark, or starts with a common question phrase, remove it
            if first_sentence.strip().endswith('?') or re.match(r'^(is it possible|can you|could you|would you|is it|are there|does|do|what|how|why|when|where|who|which)', first_sentence.strip(), re.IGNORECASE):
                return rest.lstrip()
        return answer.lstrip()

    def generate():
        # --- DEBUG LOGGING ---
        print("\n--- DEBUG: Current Chat History (stream) ---")
        print(memory.buffer)
        print("--- END DEBUG ---\n")
        # Build the full prompt as it will be sent to the LLM
        context = "[context will be filled by chain]"
        chat_history = memory.buffer
        question = user_message
        prompt_text = custom_prompt.format(context=context, chat_history=chat_history, question=question)
        enc = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(enc.encode(prompt_text))
        print(f"--- DEBUG: Total tokens sent to LLM (stream): {num_tokens} ---\n")
        # --- END DEBUG LOGGING ---
        def run_chain():
            qa_chain_stream.invoke({"question": user_message})
            handler.q.put(None)  # Signal end of stream
        thread = threading.Thread(target=run_chain)
        thread.start()
        bot_message = ""
        buffer = ""
        first_sentence_done = False
        sentence_endings = ['?', '.', '!']
        for token in handler.get_stream():
            if not first_sentence_done:
                buffer += token
                # Check if we've reached the end of the first sentence
                if any(p in buffer for p in sentence_endings):
                    # Apply the strip function to the buffer
                    cleaned = strip_leading_question_sentence(buffer)
                    first_sentence_done = True
                    if cleaned:
                        yield cleaned
            else:
                yield token
        thread.join()

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/session-chat-stream', methods=['POST'])
def session_chat_stream():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    history = get_session_history()
    # Create a ConversationSummaryMemory for this session
    session_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=1250,
        chat_memory=history
    )
    # Streaming LLM
    handler = QueueStreamHandler()
    streaming_llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", streaming=True, callbacks=[handler])
    session_chain = ConversationalRetrievalChain.from_llm(
        llm=streaming_llm,
        retriever=retriever,
        memory=session_memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

    def generate():
        bot_message = ""
        def run_chain():
            session_chain.invoke({"question": user_message})
            handler.q.put(None)  # Signal end of stream
        thread = threading.Thread(target=run_chain)
        thread.start()
        for token in handler.get_stream():
            bot_message += token
            yield token
        thread.join()
        # After streaming, store user and bot messages
        history.add_user_message(user_message)
        history.add_ai_message(bot_message)

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True) 