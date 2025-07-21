import os
from flask import Flask, render_template, request, session, redirect, url_for, Response
from lilim import Lilim
from occu import ret
import threading
import time
import requests  # Добавлен импорт для HTTP запросов
from conv import build_translation_dict

# Constants
MODEL_PATH = "./models/Jeffry/qwen3"
RAG_ENABLED = True

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = 'embedchain'
app.config['SESSION_TYPE'] = 'filesystem'

ip = "localhost"
port = 8000

# Глобальные переменные
llm = None
llm_lock = threading.Lock()
bot_name = "Lilim"

# Словарь для флагов остановки генерации (ключ - session_id)
stop_generation_flags = {}
stop_generation_lock = threading.Lock()

@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear conversation history"""
    session['chats'] = []
    session['last_status'] = ''
    if llm:
        llm.clear_history()
    return redirect(url_for('home'))

def load_model():
    """Load the LLM model once at startup"""
    global llm
    try:
        llm = Lilim(MODEL_PATH)
        success, message = llm.load_model()
        if success:
            print(f"Model loaded: {message}")
        else:
            print(f"Model loading failed: {message}")
            llm = None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        llm = None

# Load model at startup
load_model()

@app.before_request
def initialize_session():
    """Initialize session variables for every request"""
    session.setdefault('chats', [])
    session.setdefault('rag_enabled', RAG_ENABLED)
    session.setdefault('history', [])
    session.setdefault('is_processing', False)

@app.context_processor
def inject_template_vars():
    """Inject variables into templates"""
    return {
        'bot_name': bot_name,
        'chats': session.get('chats', []),
        'rag_enabled': session.get('rag_enabled', RAG_ENABLED),
        'last_status': session.get('last_status', ''),
        'is_processing': session.get('is_processing', False)
    }

def reformulate_query(original_query):
    """Improve query for better retrieval using LLM"""
    global llm
    if not llm:
        return original_query
    
    TRANSLATION_DICT = build_translation_dict()
    prompt = (
        "Time to reformulate some queies: Rephrase this query for better document retrieval. "
        "Focus on key entities and relationships. If this doeasn't make sense, look at the version with changed keyboard layout"
        "Keep it concise. Return Only the Augmented Prompt and nothing else\n\n"
        f"Original: {original_query}, changed layout: {str.translate(original_query, TRANSLATION_DICT)}\n"
        "Rephrased:"
    )
    
    print("Reformulating the prompt...")
    with llm_lock:
        return llm.generate(
            prompt, 
            think=True#, 
            #exponential_decay_length_penalty=(300, 1.1)
        )

def retrieve_context(query):
    """Retrieve relevant context from vector database"""
    try:
        results = ret(query)['Retriever']
        if not results or 'documents' not in results:
            return 'No relevant context found'
            
        context_str = ""
        for i, doc in enumerate(results["documents"]):
            context_str += f"\n\nPart {i+1}:\n"
            context_str += f"Source: {doc.meta.get('source', 'Unknown')}\n"
            context_str += f"Page: {doc.meta.get('page', 'N/A')}\n"
            context_str += f"Content: {doc.content[:500]}...\n"
        return context_str
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

def is_context_relevant(context):
    """Check if context is actually relevant to query"""
    if "No relevant context" in context or "Error retrieving" in context:
        return False
    return True

# Функция для отправки сообщения через HTTP
def add_message_to_chat(role, message):
    """Add message to chat history via HTTP request"""
    try:
        # Отправляем POST запрос к себе для добавления сообщения
        response = requests.post(
            f'http://{ip}:{port}/add_message',
            data={'role': role, 'message': message},
            timeout=None
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error adding message: {str(e)}")
        return False

@app.route("/add_message", methods=["POST"])
def add_message_endpoint():
    """Endpoint to add message to chat history"""
    role = request.form.get("role")
    message = request.form.get("message")
    if role and message:
        session['chats'].append((role, message))
        session.modified = True
        return "OK", 200
    return "Bad request", 400

# @app.route("/stop_generation", methods=["POST"])
# def stop_generation():
#     """Endpoint to stop current generation"""
#     session_id = request.cookies.get(app.session_cookie_name)
#     if session_id:
#         with stop_generation_lock:
#              stop_generation_flags[session_id] = True
#     return "OK", 200

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form["query"].strip()
        if not user_query:
            return redirect(url_for('home'))
        
        # Add user query to chat history
        session['chats'].append(("H", user_query))
        session.modified = True
        
        stream_response()
        # Process the query
        # process_query(user_query)
        
    return render_template("index.html")

#main query handler
@app.route("/stream", methods=["POST"])
def stream_response():
    user_query = request.form["query"].strip()
    # session_id = request.cookies.get(app.session_cookie_name)
    
    # # Сбрасываем флаг остановки для этой сессии
    # if session_id:
    #     with stop_generation_lock:
    #         if session_id in stop_generation_flags:
    #             del stop_generation_flags[session_id]
    
    # Добавляем запрос пользователя в историю
    add_message_to_chat("H", user_query)
    
    # 1. Получаем контекст через RAG (если включен)
    context_str = ""
    if session.get('rag_enabled', RAG_ENABLED):
        reformulated_query = reformulate_query(user_query)
        print("Reformulated query:", reformulated_query)
        if (reformulated_query == ""):
            reformulated_query = user_query
        context_str = retrieve_context(reformulated_query)
        # Add context to chat history
        # session['chats'].append(("C", context_str))
        # session.modified = True
        add_message_to_chat("C", context_str)
    
    # 2. Формируем промпт с контекстом
    augmented_prompt = user_query  # По умолчанию
    if context_str and is_context_relevant(context_str):
        augmented_prompt = (f"The query is already rephrased. You are an assisatant now." 
                            "Based on the following context:\n{context_str}\n\nAnswer this question: {user_query} DO NOT REPHRASE, ANSWER THE QUERY")
    print(augmented_prompt)
    def generate():
        try:
            # Create token stream
            token_stream = llm.generateSt(
                augmented_prompt,
                max_new_tokens=4096,
                temperature=0.7,
                think=True
            )
            start_time = time.time()
            tokens = 0
            # Stream tokens to client
            for token in token_stream:
                tokens += 1
                # Проверяем флаг остановки
                # stop_requested = False
                # if session_id:
                #     with stop_generation_lock:
                #         stop_requested = stop_generation_flags.get(session_id, False)
                
                # if stop_requested:
                #     # Отправляем сообщение об остановке
                #     yield "data: [STOPPED]\n\n"
                #     # Сбрасываем флаг
                #     with stop_generation_lock:
                #         if session_id in stop_generation_flags:
                #             del stop_generation_flags[session_id]
                #     break
                    
                if token and token != "<think>" and token != "</think>":
                    yield f"data: {token}\n\n"
            
            gen_time = time.time() - start_time
            status = f"Generated {tokens} tokens in {gen_time:.1f}s"

            session['last_status'] = status
            session['is_processing'] = False
            session.modified = True

            # Send completion marker
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Generation error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    # Return streaming response
    return Response(generate(), mimetype="text/event-stream")

@app.route("/toggle_rag", methods=["POST"])
def toggle_rag():
    """Toggle RAG functionality"""
    session['rag_enabled'] = not session.get('rag_enabled', RAG_ENABLED)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=False, host=ip, port=port)