from flask import Flask, render_template_string, request, jsonify, session, send_from_directory, Response, stream_with_context
import json
import os
import re
import uuid
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import ollama
import chromadb
from functools import wraps 
import base64
from chromadb.api.client import SharedSystemClient 
import random



    
app = Flask(__name__)
# Clear ChromaDB system cache after each request
@app.after_request
def clear_chroma_cache(response):
    try:
        SharedSystemClient.clear_system_cache()
    except Exception as e:
        app.logger.error(f"Error clearing ChromaDB cache: {str(e)}")
    return response

app.secret_key = "your_secret_key_here"

streaming_status = {}

# File paths and directories
# Add these near your other file paths and directories
PDF_STORAGE = "static/uploaded_pdfs"
PDFS_FILE = "pdfs.json"
# Add these near your other file paths and directories
IMAGE_STORAGE = "static/uploaded_images"
IMAGES_FILE = "images.json"
os.makedirs(IMAGE_STORAGE, exist_ok=True)


PROFILE_STORAGE = "static/profile_photos"
os.makedirs(PROFILE_STORAGE, exist_ok=True)

os.makedirs(PDF_STORAGE, exist_ok=True)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function



def load_user_images(user_id):
    if os.path.exists(IMAGES_FILE):
        try:
            with open(IMAGES_FILE, "r") as f:
                if os.path.getsize(IMAGES_FILE) > 0:
                    all_images = json.load(f)
                    return all_images.get(user_id, [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading images file: {e}")
    return []

def save_user_images(user_id, images):
    all_images = {}
    if os.path.exists(IMAGES_FILE):
        try:
            with open(IMAGES_FILE, "r") as f:
                if os.path.getsize(IMAGES_FILE) > 0:
                    all_images = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    all_images[user_id] = images
    with open(IMAGES_FILE, "w") as f:
        json.dump(all_images, f)

# Add a global dictionary to track PDFs per user
def load_user_pdfs(user_id):
    if os.path.exists(PDFS_FILE):
        try:
            with open(PDFS_FILE, "r") as f:
                if os.path.getsize(PDFS_FILE) > 0:
                    all_pdfs = json.load(f)
                    return all_pdfs.get(user_id, [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading PDFs file: {e}")
    return []

def save_user_pdfs(user_id, pdfs):
    all_pdfs = {}
    if os.path.exists(PDFS_FILE):
        try:
            with open(PDFS_FILE, "r") as f:
                if os.path.getsize(PDFS_FILE) > 0:
                    all_pdfs = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    all_pdfs[user_id] = pdfs
    with open(PDFS_FILE, "w") as f:
        json.dump(all_pdfs, f)

@app.route("/upload_image", methods=["POST"])
@login_required
def upload_image():
    user_id = session['user_id']
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Check if it's an image
    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return jsonify({"success": False, "error": "Only image files are allowed"}), 400
    
    original_filename = image_file.filename
    unique_filename = f"{user_id}_{uuid.uuid4()}_{original_filename}"
    file_path = os.path.join(IMAGE_STORAGE, unique_filename)
    image_file.save(file_path)
    
    images = load_user_images(user_id)
    
    # Determine the next number for the image
    if images:
        last_number = max(img.get('number', 0) for img in images)
        new_number = last_number + 1
    else:
        new_number = 1
        
   # Inside the upload_image function
    images.append({
        "path": file_path, 
        "name": original_filename, 
        "active": False,
        "number": new_number,
        "sent": False  # <-- ADD THIS LINE
    })
    
    save_user_images(user_id, images)
    
    # Return the original filename and the new number to the frontend
    return jsonify({
        "success": True, 
        "filename": original_filename,
        "imageNumber": new_number
    })

@app.route("/get_images")
@login_required 
def get_images():
    user_id = session['user_id']
    images = load_user_images(user_id)
    # Ensure the 'number' is included in the response for each image
    return jsonify({"images": [{"name": img["name"], "active": img["active"], "number": img.get("number")} for img in images]})

@app.route("/toggle_image", methods=["POST"])
@login_required 
def toggle_image():
    user_id = session['user_id']
    data = request.get_json()
    image_name = data.get("image_name")
    active = data.get("active")
    
    if image_name is None or active is None:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
    
    images = load_user_images(user_id)
    for img in images:
        if img["name"] == image_name:
            # --- START OF CHANGE ---
            if img.get("sent", False) and active:
                return jsonify({"success": False, "error": "This file has already been sent."}), 403
            # --- END OF CHANGE ---
            img["active"] = active
            save_user_images(user_id, images)
            return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Image not found"}), 404

@app.route("/delete_image", methods=["POST"])
@login_required
def delete_image():
    user_id = session['user_id']
    data = request.get_json()
    image_name = data.get("image_name")
    
    if image_name is None:
        return jsonify({"success": False, "error": "Missing image_name parameter"}), 400
    
    images = load_user_images(user_id)
    image_to_delete = next((img for img in images if img["name"] == image_name), None)
    
    if not image_to_delete:
        return jsonify({"success": False, "error": "Image not found"}), 404
    
    # Remove the file from the filesystem
    image_path = image_to_delete["path"]
    if os.path.exists(image_path):
        os.remove(image_path)
    
    # Remove the image from the user's list
    images = [img for img in images if img["name"] != image_name]
    save_user_images(user_id, images)
    
    return jsonify({"success": True})


SAVE_FILE = "conversations.json"
USERS_FILE = "users.json"
BOTS_FILE = "bots.json"
BOT_STORAGE = "static/bot_photos"
os.makedirs(BOT_STORAGE, exist_ok=True)

users = {}
if os.path.exists(USERS_FILE):
    try:
        with open(USERS_FILE, "r") as f:
            if os.path.getsize(USERS_FILE) > 0:
                users = json.load(f)
                # Ensure all users have a photo field
                for user in users.values():
                    if 'photo' not in user:
                        user['photo'] = ''
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading users file: {e}")

all_conversations = {}
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "r") as f:
        all_conversations = json.load(f)

bots = {}
if os.path.exists(BOTS_FILE):
    with open(BOTS_FILE, "r") as f:
        bots = json.load(f)

# AI Model setup
model = OllamaLLM(model="goekdenizguelmez/josiefied-qwen2.5-7b-abliterated-v2")

def format_text(text):
    return re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


# New route for user profile page
@app.route("/user-profile")
@login_required
def user_profile():
    user_id = session['user_id']
    profile_photo = f"/static/profile_photos/{users[user_id]['photo']}" if users[user_id].get('photo') else ''
    return render_template_string(user_profile_html, username=user_id, profile_photo=profile_photo)

# New route for uploading profile photo
@app.route("/upload-profile-photo", methods=['POST'])
@login_required
def upload_profile_photo():
    user_id = session['user_id']
    if 'photo' not in request.files:
        return jsonify({"success": False, "error": "No photo provided"}), 400
    
    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Remove old photo if exists
    old_photo = users[user_id].get('photo')
    if old_photo and os.path.exists(os.path.join(PROFILE_STORAGE, old_photo)):
        os.remove(os.path.join(PROFILE_STORAGE, old_photo))
    
    filename = f"{user_id}_{uuid.uuid4()}.jpg"
    photo.save(os.path.join(PROFILE_STORAGE, filename))
    users[user_id]['photo'] = filename
    
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)
    
    return jsonify({"success": True})

# Add route for serving profile photos
@app.route("/static/profile_photos/<filename>")
def profile_photo(filename):
    return send_from_directory(PROFILE_STORAGE, filename)

@app.route("/get-user-photo", methods=['GET'])
@login_required
def get_user_photo():
    user_id = session['user_id']
    photo = users[user_id].get('photo')
    if photo:
        return jsonify({"photo": f"/static/profile_photos/{photo}"})
    return jsonify({"photo": ""})

# Authentication routes
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    user = users.get(username)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({"success": False, "error": "Invalid credentials"}), 401
    session['user_id'] = username
    return jsonify({"success": True})

@app.route("/logout")
@login_required
def logout():
    session.pop('user_id', None)
    session.pop('conversation_id', None)
    return jsonify({"success": True})

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400
    if username in users:
        return jsonify({"success": False, "error": "Username exists"}), 400
    
    # FIX: Initialize the 'photo' key for new users to prevent potential KeyErrors.
    users[username] = {
        "password_hash": generate_password_hash(password),
        "photo": ""
    }
    
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)
    return jsonify({"success": True})

# Bot management routes
@app.route("/create-bot", methods=['GET', 'POST'])
@login_required
def create_bot():
    if request.method == 'POST':
        bot_data = {
            "id": str(uuid.uuid4()),
            "name": request.form['name'],
            "description": request.form['description'],
            "system_prompt": request.form['system_prompt'],
            "creator": session['user_id'],
            "photo": ""
        }
        
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename != '':
                filename = f"{bot_data['id']}_{photo.filename}"
                photo.save(os.path.join(BOT_STORAGE, filename))
                bot_data['photo'] = filename
        
        bots[bot_data['id']] = bot_data
        json.dump(bots, open(BOTS_FILE, 'w'))
        return jsonify({"success": True})
    
    return render_template_string(create_bot_html)

@app.route("/bots")
@login_required
def browse_bots():
    return render_template_string(bots_html, bots=list(bots.values()))

@app.route("/start-bot-chat/<bot_id>")
@login_required
def start_bot_chat(bot_id):
    bot = bots.get(bot_id)
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    conv_id = str(uuid.uuid4())
    user_id = session['user_id']
    
    all_conversations.setdefault(user_id, {})[conv_id] = {
        "name": f"Chat with {bot['name']}",
        "bot_id": bot_id, # Add bot_id to track the conversation's origin
        "branches": {"main": {"history": [], "parent_branch": None, "parent_message_index": None}},
        "active_branch": "main",
        "system_prompt": bot['system_prompt'],
        "bot_photo": bot.get('photo', '')  # Store bot photo in conversation
    }
    
    session["conversation_id"] = conv_id
    json.dump(all_conversations, open(SAVE_FILE, 'w'))
    return jsonify({"success": True, "conversation_id": conv_id})

# Chat routes
@app.route("/")
def home():
    if 'user_id' not in session:
        return render_template_string(login_html)
    user_id = session['user_id']
    user_conversations = all_conversations.get(user_id, {})
    return render_template_string(html_template, 
                               conversations=user_conversations,
                               username=user_id)  

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        user_id = session['user_id']
        user_conversations = all_conversations.setdefault(user_id, {})
        
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        user_message = data["message"]
        attachments = data.get("attachments", [])
        conv_id = session.get("conversation_id")

        if not conv_id or conv_id not in user_conversations:
            return jsonify({"error": "Start a conversation first"}), 400

        conversation = user_conversations[conv_id]
        active_branch = conversation.get("active_branch", "main")
        branch = conversation["branches"][active_branch]
        history = branch["history"]
        system_prompt = conversation.get("system_prompt", "")
        
        user_msg = {
            "sender": "user",
            "text": user_message,
            "versions": [user_message],
            "active_version": 0,
            "profile_photo": f"/static/profile_photos/{users[user_id]['photo']}" if users[user_id].get('photo') else '',
            # --- NEW: Save the attachments to the message object ---
            "attachments": attachments
            # --- END ---
        }
        
        unique_id = f"{random.randint(100000, 999999)}"
        bot_msg = {
            "sender": "bot",
            "text": "",
            "versions": [""],
            "active_version": 0,
            "profile_photo": f"/static/bot_photos/{conversation.get('bot_photo', '')}" if conversation.get('bot_photo') else '',
            "unique_id": unique_id
        }
        
        history.extend([user_msg, bot_msg])
        with open(SAVE_FILE, "w") as f:
            json.dump(all_conversations, f)

        streaming_status[conv_id] = {"should_stop": False}
        
        pdfs = load_user_pdfs(user_id)
        active_pdf = next((pdf for pdf in pdfs if pdf["active"]), None)
        collection = None
        rag_data = None
        
        if active_pdf:
            documents = extract_text_from_pdf(active_pdf["path"])
            if documents:
                collection = setup_knowledge_base(documents)
                if collection:
                    try:
                        response = ollama.embed(model="nomic-embed-text:latest", input=[user_message])
                        if "embeddings" in response and response["embeddings"]:
                            query_embedding = response["embeddings"][0]
                            results = collection.query(query_embeddings=[query_embedding], n_results=1)
                            if results and 'documents' in results and results['documents']:
                                rag_data = results['documents'][0][0]
                    except Exception as e:
                        app.logger.error(f"RAG query error: {str(e)}")

        images_base64 = []
        images = load_user_images(user_id)
        active_image = next((img for img in images if img["active"]), None)
        
        if active_image and os.path.exists(active_image["path"]):
            with open(active_image["path"], "rb") as img_file:
                images_base64.append(base64.b64encode(img_file.read()).decode('utf-8'))
        
        chat_context = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in history[:-2]])

        def generate():
            full_response = ""
            try:
                if images_base64:
                    messages = [{"role": "user", "content": user_message, "images": images_base64}]
                    for chunk in ollama.chat(model='gemma3:4b', messages=messages, stream=True):
                        if streaming_status.get(conv_id, {}).get("should_stop", False): break
                        token = chunk['message']['content']
                        yield token
                        full_response += token
                        history[-1]['text'] = format_text(full_response)
                        history[-1]['versions'][0] = format_text(full_response)
                elif rag_data:
                    template = f"System: {system_prompt}\n\nConversation History:\n{{history}}\n\nQuestion: {{question}}\n\nAnswer:"
                    prompt = ChatPromptTemplate.from_template(template)
                    chain = prompt | model
                    for chunk in chain.stream({"history": chat_context, "question": f"Using this data: {rag_data}. Respond to this prompt: {user_message}"}):
                        if streaming_status.get(conv_id, {}).get("should_stop", False): break
                        yield chunk
                        full_response += chunk
                        history[-1]['text'] = format_text(full_response)
                        history[-1]['versions'][0] = format_text(full_response)
                else:
                    template = f"System: {system_prompt}\n\nConversation History:\n{{history}}\n\nQuestion: {{question}}\n\nAnswer:"
                    prompt = ChatPromptTemplate.from_template(template)
                    chain = prompt | model
                    for chunk in chain.stream({"history": chat_context, "question": user_message}):
                        if streaming_status.get(conv_id, {}).get("should_stop", False): break
                        yield chunk
                        full_response += chunk
                        history[-1]['text'] = format_text(full_response)
                        history[-1]['versions'][0] = format_text(full_response)
            except Exception as e:
                history[-1]['text'] = f"Error: {str(e)}"
                history[-1]['versions'][0] = history[-1]['text']
                yield f"Error: {str(e)}"
            finally:
                bot_message = history[-1]
                
                # --- FIX START: Idempotent ID appender ---
                current_text = bot_message.get('text', '')
                stripped_text = re.sub(r'\s-\(\d{6}\)$', '', current_text).strip()
                
                # The 'text' field now only contains the clean, visible message.
                bot_message['text'] = format_text(stripped_text)
                bot_message['versions'][0] = bot_message['text']
                # --- FIX END ---

                with open(SAVE_FILE, "w") as f:
                    json.dump(all_conversations, f)
                    
                if conv_id in streaming_status:
                    del streaming_status[conv_id]
        
        return Response(stream_with_context(generate()), mimetype='text/plain')

    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return Response(f"Error: AI processing failed: {str(e)}", mimetype='text/plain', status=500)

# Add these helper functions from your RAG code
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            documents = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    documents.append(text.strip())
            return documents
    except Exception as e:
        app.logger.error(f"Error processing PDF: {str(e)}")
        return []

def setup_knowledge_base(documents):
    try:
        client = chromadb.Client()
        collection = client.create_collection(name=f"docs_{uuid.uuid4()}")
        embeddings = []
        ids = []
        docs = []
        
        for i, d in enumerate(documents):
            try:
                response = ollama.embed(model="nomic-embed-text:latest", input=[d])
                if "embeddings" in response and response["embeddings"]:
                    embedding = response["embeddings"][0]
                    embeddings.append(embedding)
                    ids.append(str(i))
                    docs.append(d)
            except Exception as e:
                app.logger.error(f"Error generating embedding for document {i}: {str(e)}")
                continue
        
        if embeddings:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=docs
            )
            return collection
        return None
    except Exception as e:
        app.logger.error(f"Error setting up knowledge base: {str(e)}")
        return None
    finally:
        # This will be handled by the after_request hook now
        pass

# Conversation management routes
@app.route("/rename_chat", methods=["POST"])
@login_required
def rename_chat():
    user_id = session['user_id']
    data = request.get_json()
    conv_id = data.get("conversation_id")
    new_name = data.get("new_name")

    if not conv_id or not new_name:
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    if user_id not in all_conversations or conv_id not in all_conversations[user_id]:
        return jsonify({"success": False, "error": "Conversation not found"}), 404
    
    all_conversations[user_id][conv_id]['name'] = new_name.strip()
    
    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    
    return jsonify({"success": True})

# Inside your Flask app (message-41.txt)

@app.route("/stop_stream", methods=["POST"])
@login_required
def stop_stream():
    user_id = session['user_id']
    conv_id = session.get("conversation_id")
    
    if not conv_id:
        return jsonify({"success": False, "error": "No active conversation"}), 400
    
    conversation = all_conversations.get(user_id, {}).get(conv_id, {})
    active_branch = conversation.get("active_branch", "main")
    history = conversation.get("branches", {}).get(active_branch, {}).get("history", [])
    
    # Bot message is last in history
    stopped_message_index = len(history) - 1
    
    if conv_id in streaming_status:
        streaming_status[conv_id]["should_stop"] = True
        return jsonify({
            "success": True,
            "stopped_message_index": stopped_message_index,
            "history": history
        })
    else:
        return jsonify({"success": False, "error": "No active stream to stop"}), 400

@app.route("/delete_chat", methods=["POST"])
@login_required
def delete_chat():
    user_id = session['user_id']
    data = request.get_json()
    conv_id = data.get("conversation_id")

    if not conv_id:
        return jsonify({"success": False, "error": "Missing conversation_id"}), 400

    user_convos = all_conversations.get(user_id, {})
    if conv_id not in user_convos:
        return jsonify({"success": False, "error": "Conversation not found"}), 404

    del user_convos[conv_id]
    if session.get('conversation_id') == conv_id:
        session.pop('conversation_id', None)

    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    
    return jsonify({"success": True})

# In your app.py, find the @app.route("/delete_message", ...) function

@app.route("/delete_message", methods=["POST"])
@login_required
def delete_message():
    user_id = session['user_id']
    data = request.get_json()
    message_index = data.get("message_index")
    conv_id = session.get("conversation_id")

    if not conv_id or message_index is None:
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    try:
        message_index = int(message_index)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid message index"}), 400

    conversation = all_conversations[user_id].get(conv_id)
    if not conversation:
        return jsonify({"success": False, "error": "Conversation not found"}), 404

    active_branch = conversation.get("active_branch", "main")
    branch = conversation["branches"][active_branch]
    history = branch["history"]
    
    # Check if index is valid
    if message_index < 0 or message_index >= len(history):
        return jsonify({"success": False, "error": "Invalid message index"}), 400

    # Delete the message
    del history[message_index]
    
    # Update all branch references that might be affected by this deletion
    # This logic remains the same as it correctly handles a single deletion
    for branch_id, branch_data in conversation["branches"].items():
        if branch_data.get("parent_message_index") is not None and branch_data["parent_message_index"] > message_index:
            branch_data["parent_message_index"] -= 1
        
        if branch_data.get("action_message_index") is not None and branch_data["action_message_index"] > message_index:
            branch_data["action_message_index"] -= 1

    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    
    # MODIFICATION: Return the updated history along with success status
    return jsonify({"success": True, "history": branch["history"]})

@app.route("/delete-bot/<bot_id>", methods=['POST'])
@login_required
def delete_bot(bot_id):
    user_id = session['user_id']
    
    if bot_id not in bots:
        return jsonify({"success": False, "error": "Bot not found"}), 404
    
    if bots[bot_id]['creator'] != user_id:
        return jsonify({"success": False, "error": "Unauthorized to delete this bot"}), 403
    
    bot_system_prompt = bots[bot_id].get('system_prompt')
    bot_photo = bots[bot_id].get('photo')
    if bot_photo:
        photo_path = os.path.join(BOT_STORAGE, bot_photo)
        if os.path.exists(photo_path):
            os.remove(photo_path)
    
    del bots[bot_id]
    with open(BOTS_FILE, 'w') as f:
        json.dump(bots, f)
    
    for user_id, user_convos in list(all_conversations.items()):
        convos_to_delete = []
        for conv_id, conv_data in user_convos.items():
            if conv_data.get('system_prompt') == bot_system_prompt:
                convos_to_delete.append(conv_id)
        for conv_id in convos_to_delete:
            del user_convos[conv_id]
    
    with open(SAVE_FILE, 'w') as f:
        json.dump(all_conversations, f)
    
    return jsonify({"success": True})

@app.route("/conversations")
@login_required
def conversations():
    user_id = session['user_id']
    # Also send bot info needed for grouping conversations in the UI
    bot_info = {bot_id: {"name": data["name"]} for bot_id, data in bots.items()}
    return jsonify({
        "conversations": all_conversations.get(user_id, {}),
        "bots": bot_info
    })

@app.route("/load_conversation", methods=["POST"])
@login_required
def load_conversation():
    data = request.get_json()
    conv_id = data.get("conversation_id")
    session["conversation_id"] = conv_id
    return jsonify({"success": True})

# Static files
@app.route("/static/bot_photos/<filename>")
def bot_photo(filename):
    return send_from_directory(BOT_STORAGE, filename)

@app.route("/regenerate_message", methods=["POST"])
@login_required
def regenerate_message():
    try:
        user_id = session['user_id']
        data = request.json
        message_index = data.get("message_index")
        conv_id = session.get("conversation_id")

        if not conv_id:
            return jsonify({"error": "No active conversation"}), 400

        conversation = all_conversations.get(user_id, {}).get(conv_id)
        active_branch = conversation.get("active_branch", "main")
        history = conversation["branches"][active_branch]["history"]
        message_index = int(message_index)

        if message_index < 0 or message_index >= len(history):
            return jsonify({"error": "Invalid message index"}), 400

        target_message = history[message_index]
        if target_message["sender"] != "bot":
            return jsonify({"error": "Can only regenerate bot messages"}), 400

        user_message = next(
            (msg["text"] for msg in reversed(history[:message_index]) if msg["sender"] == "user"), None
        )

        if not user_message:
            return jsonify({"error": "No user message found to regenerate from"}), 400

        new_branch_id = ""
        if active_branch == "main":
            b_count = sum(1 for bid in conversation["branches"] if re.match(r'^B\d+$', bid))
            new_branch_id = f"B{b_count + 1}"
        else:
            base_prefix = active_branch.split('-')[0]
            sub_b_count = sum(1 for bid in conversation["branches"] if bid.startswith(f"{base_prefix}-B"))
            new_branch_id = f"{base_prefix}-B{sub_b_count + 1}"
            
        new_branch_history = [dict(msg) for msg in history[:message_index]]

        system_prompt = conversation.get("system_prompt", "")
        template = f"System: {system_prompt}\n\nConversation History:\n{{history}}\n\nQuestion: {{question}}\n\nAnswer:"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        chat_context = "\n".join(
            f"{msg['sender']}: {msg['text']}" for msg in new_branch_history if msg["sender"] in ["user", "bot"]
        )
        pdfs = load_user_pdfs(user_id)
        active_pdf = next((pdf for pdf in pdfs if pdf["active"]), None)
        collection = None
        
        if active_pdf:
            documents = extract_text_from_pdf(active_pdf["path"])
            if documents:
                collection = setup_knowledge_base(documents)
        streaming_status[conv_id] = {"should_stop": False}

        def generate():
            full_response = ""
            unique_id = f"{random.randint(100000, 999999)}"
            try:
                if collection:
                    response = ollama.embed(model="nomic-embed-text:latest", input=[user_message])
                    query_embedding = response["embeddings"][0]
                    results = collection.query(query_embeddings=[query_embedding], n_results=1)
                    data = results['documents'][0][0]
                    for chunk in chain.stream({"history": chat_context, "question": f"Using this data: {data}. Respond to this prompt: {user_message}"}):
                        if streaming_status.get(conv_id, {}).get("should_stop", False): break
                        yield chunk
                        full_response += chunk
                else:
                    for chunk in chain.stream({"history": chat_context, "question": user_message}):
                        if streaming_status.get(conv_id, {}).get("should_stop", False): break
                        yield chunk
                        full_response += chunk
            except Exception as e:
                yield f"Error: {str(e)}"
            finally:
                                # --- FIX START: Store ID separately, not in text ---
                stripped_response = re.sub(r'\s-\(\d{6}\)$', '', full_response).strip()
                formatted_response = format_text(stripped_response)
                # --- FIX END ---
                
                bot_photo = conversation.get("bot_photo", "")
                new_bot_message = {
                    "sender": "bot", "text": formatted_response, # Use the clean response
                    "versions": [target_message["text"], formatted_response], "active_version": 1,
                    "profile_photo": f"/static/bot_photos/{bot_photo}" if bot_photo else '',
                    "unique_id": unique_id
                }
                new_branch_history.append(new_bot_message)
                
                conversation["branches"][new_branch_id] = {
                    "history": new_branch_history, "parent_branch": active_branch,
                    "parent_message_index": message_index - 1, "action_message_index": message_index
                }
                conversation["active_branch"] = new_branch_id

                with open(SAVE_FILE, "w") as f:
                    json.dump(all_conversations, f)

                if conv_id in streaming_status:
                    del streaming_status[conv_id]

        return Response(stream_with_context(generate()), mimetype='text/plain')

    except Exception as e:
        app.logger.error(f"Regeneration error: {str(e)}")
        return Response(f"Error: Regeneration failed: {str(e)}", mimetype='text/plain', status=500)
    
@app.route("/set_message_version", methods=["POST"])   
@login_required
def set_message_version():
    user_id = session['user_id']
    data = request.get_json()
    message_index = data.get("message_index")
    version_index = data.get("version_index")
    conv_id = session.get("conversation_id")

    if not conv_id or message_index is None or version_index is None:
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    try:
        message_index = int(message_index)
        version_index = int(version_index)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid indices"}), 400

    conversation = all_conversations.get(user_id, {}).get(conv_id)
    active_branch = conversation.get("active_branch", "main")
    history = conversation["branches"][active_branch]["history"]

    message = history[message_index]
    if "versions" not in message or version_index < 0 or version_index >= len(message["versions"]):
        return jsonify({"success": False, "error": "Invalid version index"}), 400

    message["active_version"] = version_index
    message["text"] = message["versions"][version_index]

    new_bot_text = None
    if message["sender"] == "user" and message_index + 1 < len(history) and history[message_index + 1]["sender"] == "bot":
        system_prompt = conversation.get("system_prompt", "")
        template = f"""System: {system_prompt}
        
        Conversation History:
        {{history}}
        
        Question: {{question}}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        chat_context = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in history[:message_index]])
        response = chain.invoke({
            "history": chat_context,
            "question": message["text"]
        })
        new_bot_text = format_text(response)
        history[message_index + 1]["text"] = new_bot_text
        history[message_index + 1]["versions"] = [new_bot_text]
        history[message_index + 1]["active_version"] = 0

    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)

    return jsonify({
        "success": True, 
        "new_text": message["text"],
        "new_bot_text": new_bot_text
    })

# --- NEW SECTION: Branching Functionality ---
@app.route("/edit_message", methods=["POST"])
@login_required
def edit_message():
    user_id = session['user_id']
    data = request.get_json()
    message_index = data.get("message_index")
    new_text = data.get("new_text")
    conv_id = session.get("conversation_id")

    if not conv_id or message_index is None or not new_text:
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    try:
        message_index = int(message_index)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid message index"}), 400

    user_convos = all_conversations.get(user_id, {})
    conversation = user_convos.get(conv_id)
    if not conversation:
        return jsonify({"success": False, "error": "Conversation not found"}), 404
    
    active_branch = conversation.get("active_branch", "main")
    history = conversation["branches"][active_branch]["history"]
    
    if message_index < 0 or message_index >= len(history):
        return jsonify({"success": False, "error": "Invalid message index"}), 400

    message = history[message_index]
    if message['sender'] != 'user':
        return jsonify({"success": False, "error": "Cannot edit bot message"}), 400

    u_count = sum(1 for bid in conversation["branches"] if re.match(r'^U\d+$', bid))
    new_branch_id = f"U{u_count + 1}"
    
    new_branch_history = [dict(msg) for msg in history[:message_index]]
    new_message = {
        "sender": "user",
        "text": new_text.strip(),
        "versions": [message["text"], new_text.strip()] if message.get("versions") else [new_text.strip()],
        "active_version": len(message.get("versions", []))
    }
    
    pdfs = load_user_pdfs(user_id)
    active_pdf = next((pdf for pdf in pdfs if pdf["active"]), None)
    collection = None
    
    if active_pdf:
        documents = extract_text_from_pdf(active_pdf["path"])
        if documents:
            collection = setup_knowledge_base(documents)
    
    system_prompt = conversation.get("system_prompt", "")
    template = f"System: {system_prompt}\n\nConversation History:\n{{history}}\n\nQuestion: {{question}}\n\nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    chat_context = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in new_branch_history])
    streaming_status[conv_id] = {"should_stop": False}

    def generate():
        full_response = ""
        unique_id = f"{random.randint(100000, 999999)}"
        try:
            if collection:
                response = ollama.embed(model="nomic-embed-text:latest", input=[new_text])
                query_embedding = response["embeddings"][0]
                results = collection.query(query_embeddings=[query_embedding], n_results=1)
                data = results['documents'][0][0]
                for chunk in chain.stream({"history": chat_context, "question": f"Using this data: {data}. Respond to this prompt: {new_text}"}):
                    if streaming_status.get(conv_id, {}).get("should_stop", False): break
                    yield chunk
                    full_response += chunk
            else:
                for chunk in chain.stream({ "history": chat_context, "question": new_text }):
                    if streaming_status.get(conv_id, {}).get("should_stop", False): break
                    yield chunk
                    full_response += chunk
        except Exception as e:
            yield f"Error: {str(e)}"
        finally:
                        # --- FIX START: Store ID separately, not in text ---
            stripped_response = re.sub(r'\s-\(\d{6}\)$', '', full_response).strip()
            formatted_response = format_text(stripped_response)
            # --- FIX END ---

            new_branch_history.append(new_message)
            new_branch_history.append({
                "sender": "bot",
                "text": formatted_response, # Use the clean response
                "versions": [formatted_response],
                "active_version": 0,
                "unique_id": unique_id
            })
            
            conversation["branches"][new_branch_id] = {
                 "history": new_branch_history,
                "parent_branch": active_branch,
                "parent_message_index": message_index,
                "action_message_index": message_index
            }
            conversation["active_branch"] = new_branch_id
            
            with open(SAVE_FILE, "w") as f:
                json.dump(all_conversations, f)
        
            if conv_id in streaming_status: del streaming_status[conv_id]

    return Response(stream_with_context(generate()), mimetype='text/plain')
# --- NEW SECTION: Branch Switching ---
@app.route("/switch_branch", methods=["POST"])
@login_required
def switch_branch():
    user_id = session['user_id']
    data = request.get_json()
    conv_id = session.get("conversation_id")
    branch_id = data.get("branch_id")

    if not conv_id or not branch_id:
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    user_convos = all_conversations.get(user_id, {})
    conversation = user_convos.get(conv_id)
    if not conversation or branch_id not in conversation["branches"]:
        return jsonify({"success": False, "error": "Invalid branch"}), 404

    conversation["active_branch"] = branch_id
    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    
    # Return only the history of the newly active branch
    return jsonify({
        "success": True,
        "history": conversation["branches"][branch_id]["history"]
    })

# --- CHANGED SECTION: Ensure /history returns only the active branch's history ---
@app.route("/history")
@login_required
def history():
    user_id = session['user_id']
    conv_id = session.get("conversation_id")
    if not conv_id:
        return jsonify({"history": [], "bot_name": "Omni.ai"})
    conversation = all_conversations.get(user_id, {}).get(conv_id, {})
    active_branch = conversation.get("active_branch", "main")
    bot_name_match = re.match(r"Chat with (.+)", conversation.get("name", "Chat with Omni.ai"))
    bot_name = bot_name_match.group(1) if bot_name_match else "Omni.ai"
    history_data = conversation.get("branches", {"main": {"history": []}})[active_branch]["history"]
    
    # Add profile photos to messages
    for msg in history_data:
        if msg["sender"] == "user":
            msg["profile_photo"] = f"/static/profile_photos/{users[user_id]['photo']}" if users[user_id].get('photo') else ''
        elif msg["sender"] == "bot":
            # Find the bot associated with this conversation
            bot_id = next((bid for bid, b in bots.items() if b['system_prompt'] == conversation.get('system_prompt')), None)
            msg["profile_photo"] = f"/static/bot_photos/{bots[bot_id]['photo']}" if bot_id and bots[bot_id].get('photo') else ''
    
    return jsonify({
        "history": history_data,
        "bot_name": bot_name
    })

# --- NEW SECTION: Get Branches ---
# Find this function in your Python code
@app.route("/branches")
@login_required
def get_branches():
    user_id = session['user_id']
    conv_id = session.get("conversation_id")
    if not conv_id:
        # Add conversation_id: None for consistency
        return jsonify({"branches": {}, "active_branch": "main", "conversation_id": None})
    conversation = all_conversations.get(user_id, {}).get(conv_id, {})
    return jsonify({
        "branches": {k: {"parent_branch": v["parent_branch"], 
                         "parent_message_index": v["parent_message_index"],
                         "action_message_index": v.get("action_message_index")} 
                    for k, v in conversation.get("branches", {"main": {"history": []}}).items()},
        "active_branch": conversation.get("active_branch", "main"),
        "conversation_id": conv_id  # <--- ADD THIS LINE
    })

@app.route("/new_chat_with_bot", methods=["POST"])
@login_required
def new_chat_with_bot():
    try:
        user_id = session['user_id']
        data = request.json
        bot_name = data.get("bot_name")
        
        if not bot_name:
            return jsonify({"error": "Bot name required"}), 400

        # Find the bot by its name to get its ID and data
        bot_info = next(((bot_id, bot_data) for bot_id, bot_data in bots.items() if bot_data['name'] == bot_name), None)
        
        if not bot_info:
            return jsonify({"error": "Bot not found"}), 404
        
        bot_id, bot = bot_info

        # Ensure user has a conversation dict
        user_convos = all_conversations.setdefault(user_id, {})

        # Create the new conversation with bot_id and bot_photo
        new_conversation_id = str(uuid.uuid4())
        user_convos[new_conversation_id] = {
            "name": f"Chat with {bot['name']}",
            "bot_id": bot_id, # Add bot_id to track the conversation's origin
            "branches": {
                "main": {
                    "history": [],
                    "parent_branch": None,
                    "parent_message_index": None
                }
            },
            "active_branch": "main",
            "system_prompt": bot['system_prompt'],
            "bot_photo": bot.get('photo', '')
        }

        # Switch into the new convo
        session["conversation_id"] = new_conversation_id

        # Persist to disk
        with open(SAVE_FILE, "w") as f:
            json.dump(all_conversations, f)

        return jsonify({"success": True, "conversation_id": new_conversation_id}), 200

    except Exception as e:
        app.logger.error(f"New chat error: {str(e)}")
        return jsonify({"error": f"Failed to create new chat: {str(e)}"}), 500


# New Routes for RAG
@app.route("/upload_pdf", methods=["POST"])
@login_required
def upload_pdf():
    user_id = session['user_id']
    if 'pdf' not in request.files:
        return jsonify({"success": False, "error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400
    
    filename = f"{user_id}_{uuid.uuid4()}_{pdf_file.filename}"
    file_path = os.path.join(PDF_STORAGE, filename)
    pdf_file.save(file_path)
    
    pdfs = load_user_pdfs(user_id)
    
    # --- START OF CHANGES ---
    # Determine the next number for the PDF
    if pdfs:
        # Use .get('number', 0) to handle older entries without a number
        last_number = max(pdf.get('number', 0) for pdf in pdfs)
        new_number = last_number + 1
    else:
        new_number = 1
        
    # Add the new PDF with its calculated number
    # Inside the upload_pdf function
    pdfs.append({
        "path": file_path, 
        "name": pdf_file.filename, 
        "active": False,
        "number": new_number,
        "sent": False  # <-- ADD THIS LINE
    })
    
    try:
        save_user_pdfs(user_id, pdfs)
        app.logger.info(f"PDF uploaded for user {user_id}: {pdf_file.filename}")
    except IOError as e:
        app.logger.error(f"Failed to save PDFs: {e}")
        return jsonify({"success": False, "error": "Failed to save PDF data"}), 500
    
    # --- MODIFIED RETURN VALUE ---
    return jsonify({
        "success": True, 
        "filename": pdf_file.filename,
        "pdfNumber": new_number
    })

@app.route("/get_pdfs")
@login_required
def get_pdfs():
    user_id = session['user_id']
    pdfs = load_user_pdfs(user_id)
    # --- MODIFIED LIST COMPREHENSION ---
    # Ensure the 'number' is included in the response for each PDF
    return jsonify({"pdfs": [{"name": pdf["name"], "active": pdf["active"], "number": pdf.get("number")} for pdf in pdfs]})

@app.route("/toggle_pdf", methods=["POST"])
@login_required
def toggle_pdf():
    user_id = session['user_id']
    data = request.get_json()
    pdf_name = data.get("pdf_name")
    active = data.get("active")
    
    if pdf_name is None or active is None:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
    
    pdfs = load_user_pdfs(user_id)
    for pdf in pdfs:
        if pdf["name"] == pdf_name:
            # --- START OF CHANGE ---
            if pdf.get("sent", False) and active:
                return jsonify({"success": False, "error": "This file has already been sent."}), 403
            # --- END OF CHANGE ---
            pdf["active"] = active
            save_user_pdfs(user_id, pdfs)
            return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "PDF not found"}), 404

@app.route("/delete_pdf", methods=["POST"])
@login_required
def delete_pdf():
    user_id = session['user_id']
    data = request.get_json()
    pdf_name = data.get("pdf_name")
    
    if pdf_name is None:
        return jsonify({"success": False, "error": "Missing pdf_name parameter"}), 400
    
    pdfs = load_user_pdfs(user_id)
    pdf_to_delete = next((pdf for pdf in pdfs if pdf["name"] == pdf_name), None)
    
    if not pdf_to_delete:
        return jsonify({"success": False, "error": "PDF not found"}), 404
    
    # Remove the file from the filesystem
    pdf_path = pdf_to_delete["path"]
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    # Remove the PDF from the user's list
    pdfs = [pdf for pdf in pdfs if pdf["name"] != pdf_name]
    save_user_pdfs(user_id, pdfs)
    
    return jsonify({"success": True})


@app.route("/mark_files_as_sent", methods=["POST"])
@login_required
def mark_files_as_sent():
    user_id = session['user_id']
    data = request.get_json()
    filenames_to_mark = data.get("filenames", [])

    if not filenames_to_mark:
        return jsonify({"success": True})

    # Mark images as sent
    images = load_user_images(user_id)
    for image in images:
        if image['name'] in filenames_to_mark:
            image['sent'] = True
            image['active'] = False # Ensure it's deactivated
    save_user_images(user_id, images)

    # Mark PDFs as sent
    pdfs = load_user_pdfs(user_id)
    for pdf in pdfs:
        if pdf['name'] in filenames_to_mark:
            pdf['sent'] = True
            pdf['active'] = False # Ensure it's deactivated
    save_user_pdfs(user_id, pdfs)

    return jsonify({"success": True})

#new routes for bot management

@app.route("/my-bots")
@login_required
def my_bots():
    user_id = session['user_id']
    user_bots = {bot_id: bot_data for bot_id, bot_data in bots.items() if bot_data['creator'] == user_id}
    return render_template_string(my_bots_html, my_bots=user_bots.values())

@app.route("/edit-bot/<bot_id>", methods=['POST'])
@login_required
def edit_bot(bot_id):
    user_id = session['user_id']
    if bot_id not in bots or bots[bot_id]['creator'] != user_id:
        return jsonify({"success": False, "error": "Bot not found or unauthorized"}), 403

    bot_data = bots[bot_id]
    bot_data["name"] = request.form['name']
    bot_data["description"] = request.form['description']
    bot_data["system_prompt"] = request.form['system_prompt']

    if 'photo' in request.files and request.files['photo'].filename != '':
        photo = request.files['photo']
        old_photo = bot_data.get('photo')
        if old_photo and os.path.exists(os.path.join(BOT_STORAGE, old_photo)):
            os.remove(os.path.join(BOT_STORAGE, old_photo))
        
        filename = f"{bot_id}_{photo.filename}"
        photo.save(os.path.join(BOT_STORAGE, filename))
        bot_data['photo'] = filename

    with open(BOTS_FILE, 'w') as f:
        json.dump(bots, f)
    
    return jsonify({"success": True})


# Add these new routes after the existing authentication routes

@app.route("/change-username", methods=["POST"])
@login_required
def change_username():
    user_id = session['user_id']
    data = request.get_json()
    new_username = data.get("new_username")
    
    if not new_username:
        return jsonify({"success": False, "error": "New username required"}), 400
    
    if new_username in users and new_username != user_id:
        return jsonify({"success": False, "error": "Username already taken"}), 400
    
    # Update user data
    users[new_username] = users.pop(user_id)
    session['user_id'] = new_username
    
    # Update all conversations
    if user_id in all_conversations:
        all_conversations[new_username] = all_conversations.pop(user_id)
    
    # Update bot creators
    for bot in bots.values():
        if bot['creator'] == user_id:
            bot['creator'] = new_username
    
    # Update PDFs
    pdfs = load_user_pdfs(user_id)
    if pdfs:
        save_user_pdfs(new_username, pdfs)
        if os.path.exists(PDFS_FILE):
            try:
                with open(PDFS_FILE, "r") as f:
                    all_pdfs = json.load(f)
                if user_id in all_pdfs:
                    all_pdfs[new_username] = all_pdfs.pop(user_id)
                with open(PDFS_FILE, "w") as f:
                    json.dump(all_pdfs, f)
            except (json.JSONDecodeError, IOError):
                pass
    
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)
    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    with open(BOTS_FILE, "w") as f:
        json.dump(bots, f)
    
    return jsonify({"success": True, "new_username": new_username})

@app.route("/delete-account", methods=["POST"])
@login_required
def delete_account():
    user_id = session['user_id']
    
    # Remove user's profile photo
    if users[user_id].get('photo'):
        photo_path = os.path.join(PROFILE_STORAGE, users[user_id]['photo'])
        if os.path.exists(photo_path):
            os.remove(photo_path)
    
    # Remove user's PDFs
    pdfs = load_user_pdfs(user_id)
    for pdf in pdfs:
        if os.path.exists(pdf["path"]):
            os.remove(pdf["path"])
    if os.path.exists(PDFS_FILE):
        try:
            with open(PDFS_FILE, "r") as f:
                all_pdfs = json.load(f)
            if user_id in all_pdfs:
                del all_pdfs[user_id]
            with open(PDFS_FILE, "w") as f:
                json.dump(all_pdfs, f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Remove user's bots and their photos
    bots_to_delete = [bot_id for bot_id, bot in bots.items() if bot['creator'] == user_id]
    for bot_id in bots_to_delete:
        if bots[bot_id].get('photo'):
            photo_path = os.path.join(BOT_STORAGE, bots[bot_id]['photo'])
            if os.path.exists(photo_path):
                os.remove(photo_path)
        del bots[bot_id]
    
    # Remove user's conversations
    if user_id in all_conversations:
        del all_conversations[user_id]
    
    # Remove user
    del users[user_id]
    session.pop('user_id', None)
    session.pop('conversation_id', None)
    
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)
    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    with open(BOTS_FILE, "w") as f:
        json.dump(bots, f)
    
    return jsonify({"success": True})




# --- CHANGED SECTION: Completely Rewritten HTML Template with Branch Support ---
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
    <style>
        body { 
            margin: 0; 
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; 
            display: flex; 
            height: 100vh; 
            background: #000; 
            color: #fff; 
            overflow: hidden;
        }
        
        .sidebar { 
            width: 260px; 
            background: #1a1a1a; 
            padding: 16px; 
            border-right: 1px solid #333; 
            display: flex; 
            flex-direction: column; 
            position: fixed; 
            top: 0;
            left: 0;
            height: 100%;
            z-index: 20;
            transform: translateX(-100%);
            transition: transform 0.3s ease;
            overflow-y: auto;
        }
        
        .sidebar.open {
            transform: translateX(0);
        }
        
        .chat-area { 
            flex: 1; 
            display: flex; 
            flex-direction: column; 
            background: #000;
            position: relative;
        }
        
        .chat-header { 
            background: #1a1a1a; 
            padding: 20px; 
            font-size: 16px; 
            font-weight: 600; 
            border-bottom: 1px solid #333; 
            display: flex; 
            align-items: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 10;
            height: 61px; /* Fixed height for consistent spacing */
        }
        
        .chat-messages { 
            flex: 1; 
            overflow-y: auto; 
            padding: 20px; 
            background: #000; 
            margin-top: 61px; /* Matches chat-header height */
            margin-bottom: 80px; /* Space for input box */
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 0; /* Prevents overflow */
            scroll-behavior: smooth; /* Optional for smooth scrolling */
            overflow-anchor: none; /* Prevents automatic scrolling */
        }
        
        .message { 
            margin: 24px 0; 
            padding: 12px 16px; 
            border-radius: 8px; 
            max-width: 800px;
            width: calc(100% - 40px);
            line-height: 1.5; 
            font-size: 15px; 
            position: relative;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .message.menu-active {
    z-index: 10;
}

        
      .user-message {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px 12px 0 12px;
    color: #fff;
    max-width: 60%;            /* much narrower */
    margin-left: auto;         /* right align */
    margin-right: 20px;        /* small right gap */
    padding: 8px 10px;         /* tighter padding */
    font-size: 14px;
    line-height: 1.3;
    display: flex;             /* avatar on far right */
    align-items: flex-start;
    gap: 8px;
}
.user-message .profile-pic {
    order: 2;                  /* avatar after text */
}
        
       .bot-message {
  background: transparent;
  border: none;
  border-radius: 12px;
  max-width: 800px;
  width: 100%;
  margin: 0 auto 24px;
  padding: 0;
}
.bot-message .message-box {
  background: rgba(42, 42, 42, .45);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255, 255, 255, .08);
  border-radius: 12px;
  padding: 12px 16px;
}
.bot-message .message-controls {
  display: flex;
  gap: 8px;
  justify-content: center;
  margin-top: 6px;
  opacity: 0;
  transition: opacity .2s ease;
}
.bot-message:hover .message-controls {
  opacity: 1;
}
.bot-message .message-controls button {
  background: rgba(68, 68, 68, .75);
  border: none;
  color: #fff;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 14px;
  cursor: pointer;
}
.bot-message .message-controls button:hover {
  background: rgba(85, 85, 85, 1);
}
        
        /* ---- chat bar ---- */
.input-box {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  width: calc(100% - 40px);
  max-width: 800px;
  min-height: 60px;               /* taller base */
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 14px;
  box-shadow: 0 2px 12px rgba(0,0,0,.35);
  z-index: 15;
}

#chat-messages {
  margin-left: auto;
  margin-right: auto;
  max-width: 800px;   /* same as input-box */
  width: 100%;
}

.bar-btn {
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 10px;
  background: #444;
  color: #bbb;
  font-size: 24px;          /* slightly larger */
  font-family: Arial, sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background .2s;
  
  line-height: -5; /* Add this to fix vertical centering */
}
.bar-btn:hover { background: #555; }



textarea {
  flex: 1;
  max-height: 120px;              /* grow up to 5 lines */
  min-height: 40px;
  padding: 0;
  background: transparent;
  border: none;
  color: #eee;
  font-size: 15px;
  resize: none;
  outline: none;
  line-height: 1.4;
  overflow-y: auto;
}

/* ---- popup (no changes) ---- */
.files-popup {
  position: absolute;
  bottom: 68px;
  left: 0;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 10px;
  width: 220px;
  display: none;
  flex-direction: column;
  z-index: 1000;
}
.popup-header {
  background: none;
  border: none;
  color: #eee;
  padding: 8px 12px;
  text-align: left;
  cursor: pointer;
  font-size: 14px;
  border-bottom: 1px solid #333;
}
.popup-header:hover { background: #3a3a3a; }
.popup-content { padding: 8px; }
.popup-content .file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  color: #ddd;
  margin-bottom: 4px;
}
.popup-content .file-item input[type="checkbox"] { margin-right: 6px; }
.popup-content .file-item .del {
  border: none;
  background: none;
  color: #ff6666;
  cursor: pointer;
}

#add-files-btn,
#send-stop-btn {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  border: none;
  background: #444;
  color: #bbb;
  border-radius: 8px;
  cursor: pointer;
  display: grid;
  place-items: center;
  transition: background .2s;
  position: relative; /* Allows fine-tuning */
  top: 1px; /* Minor vertical adjustment */
  font-family: Arial, sans-serif; /* Consistent rendering */
  font-weight: bold; /* Makes + thicker */
}

#add-files-btn:hover,
#send-stop-btn:hover { background: #555; }
textarea {
  flex: 1;
  height: 36px;
  background: transparent;
  border: none;
  color: #eee;
  font-size: 15px;
  resize: none;
  outline: none;
  line-height: 36px;
}

/* popup */
.files-popup {
  position: absolute;
  bottom: 60px;
  left: 0;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 8px;
  padding: 4px 0;
  min-width: 160px;
  display: none;
  flex-direction: column;
  z-index: 1000;
}
.files-popup .choice {
  background: none;
  border: none;
  color: #eee;
  padding: 8px 12px;
  text-align: left;
  cursor: pointer;
}
.files-popup .choice:hover { background: #3a3a3a; }

/* collapsible lists */
.file-list {
  border-top: 1px solid #444;
  margin-top: 4px;
  padding-top: 4px;
}
.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 12px;
  font-size: 13px;
  color: #ddd;
}
.file-item input[type="checkbox"] { margin-right: 6px; }
.file-item .del {
  border: none;
  background: none;
  color: #ff6666;
  cursor: pointer;
}

        
        .input-box.disabled {
            pointer-events: none;
            opacity: 0.7;
        }
        
        textarea { 
    flex: 1; 
    padding: 12px 20px; 
    border: none; 
    background: transparent; 
    color: #fff; 
    font-size: 15px; 
    outline: none; 
    resize: none; 
    min-height: 44px; 
    max-height: 200px; 
    line-height: 1.5; 
    transition: height 0.2s; 
    overflow-y: auto; /* Enable vertical scrolling */
    scrollbar-width: thin; /* Firefox: slim scrollbar */
    scrollbar-color: #555 #1a1a1a; /* Firefox: scrollbar colors */
}

/* Webkit browsers (Chrome, Safari) scrollbar styling */
textarea::-webkit-scrollbar {
    width: 6px;
}
textarea::-webkit-scrollbar-track {
    background: #1a1a1a;
}
textarea::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 3px;
}
textarea::-webkit-scrollbar-thumb:hover {
    background: #666;
}
        
        button { 
            padding: 12px 24px; 
            background: #333; 
            border: none; 
            border-radius: 20px; 
            color: #fff; 
            cursor: pointer; 
            font-weight: 500; 
            transition: background 0.2s; 
            height: 44px; 
        }
        
        button:hover { 
            background: #444; 
        }
        
        #send-stop-btn {
            width: 44px;
            height: 44px;
            min-width: 44px;
            border-radius: 50%;
            padding: 0;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        
        #sidebar-toggle-header {
            background: #2b2b2b;
            padding: 12px 16px;
            border-radius: 8px;
            margin-right: 20px;
            text-align: center;
            cursor: pointer;
        }
        
        #sidebar-toggle {
            background: #2b2b2b;
            padding: 12px 16px;
            border-radius: 8px;
            margin-left: 10px;
            margin-bottom: 10px;
            text-align: center;
            cursor: pointer;
            width: 44px;
        }
        
        .sidebar-links {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .sidebar-links a {
            color: #aaa;
            text-decoration: none;
            padding: 10px;
            border-radius: 8px;
            transition: all 0.2s;
        }
        
        .sidebar-links a:hover {
            background: #2b2b2b;
            color: #fff;
        }
        
        .conversation-item { 
            padding: 8px 12px; 
            cursor: pointer; 
            border-radius: 4px; 
            margin: 2px 0; 
            font-size: 14px; 
            color: #aaa; 
            transition: background 0.2s; 
        }
        
        .conversation-item:hover { 
            background: #2b2b2b; 
            color: #fff; 
        }
        
        .delete-convo-btn { 
            position: absolute; 
            right: 8px; 
            top: 50%; 
            transform: translateY(-50%); 
            background: transparent; 
            border: none; 
            color: #ff4444; 
            cursor: pointer; 
            opacity: 0; 
            transition: opacity 0.2s; 
            font-size: 14px; 
        }
        
        .conversation-item:hover .delete-convo-btn { 
            opacity: 1; 
        }
        
        .logout-btn { 
            background: #ff4444; 
            padding: 12px 16px; 
            border-radius: 8px; 
            cursor: pointer; 
            transition: background 0.2s; 
            border: none; 
            color: #fff; 
            position: absolute; 
            bottom: 16px; 
            left: 16px; 
            right: 16px; 
        }
        
        .logout-btn:hover { 
            background: #cc3333; 
        }
        
        .message strong { 
            display: block; 
            margin-bottom: 6px; 
            font-size: 13px; 
            color: #888; 
        }
        
        .message-controls { 
            display: flex; 
            gap: 8px; 
            margin-top: 8px; 
            justify-content: flex-end;
        }
        
        .regenerate-btn, .edit-message-btn, .branch-btn { 
            background: #333; 
            border: 1px solid #444; 
            color: #fff; 
            border-radius: 6px; 
            width: 32px; 
            height: 32px; 
            cursor: pointer; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-size: 16px; 
            transition: background 0.2s, border-color 0.2s;
        }
        
        .regenerate-btn:hover, .edit-message-btn:hover, .branch-btn:hover { 
            background: #444; 
            border-color: #555; 
            color: #fff;
        }
        
        .delete-message-btn { 
            background: #ff4444; 
            border: 1px solid #ff6666; 
            color: #fff; 
            border-radius: 6px; 
            width: 32px; 
            height: 32px; 
            cursor: pointer; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            transition: background 0.2s, border-color 0.2s;
        }
        
        .delete-message-btn:hover { 
            background: #ff6666; 
            border-color: #ff8888;
        }
        
        .loading-spinner { 
            display: inline-block; 
            width: 16px; 
            height: 16px; 
            border: 2px solid rgba(255, 255, 255, 0.3); 
            border-radius: 50%; 
            border-top-color: #fff; 
            animation: spin 1s ease-in-out infinite; 
        }
        
        @keyframes spin { 
            to { transform: rotate(360deg); } 
        }
        
        .branch-menu {
            position: absolute;
            background: #222;
            border: 1px solid #333;
            border-radius: 4px;
            right: 0;
            z-index: 1000;
            min-width: 150px;
        }
        
        .branch-item {
            padding: 8px 12px;
            cursor: pointer;
            color: #aaa;
        }
        
        .branch-item:hover {
            background: #2b2b2b;
            color: #fff;
        }
        
        .branch-item.active {
            background: #2b2b2b;
            color: #fff;
        }
        
        .bot-group {
            margin-bottom: 16px;
        }
        
        .bot-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #222;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            color: #fff;
        }
        
        .bot-header:hover {
            background: #2b2b2b;
        }
        
        .bot-conversations {
            display: none;
            padding-left: 12px;
        }
        
        .new-chat-bot-btn {
            background: transparent;
            border: none;
            color: #aaa;
            cursor: pointer;
            font-size: 14px;
            padding: 0 8px;
        }
        
        .new-chat-bot-btn:hover {
            color: #fff;
        }
        
        .code-block-container {
            position: relative;
            margin: 8px 0;
        }
        
        .code-block-container pre {
            background: #2d2d2d;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', Courier, monospace;
            font-size: 14px;
            color: #ddd;
            margin: 0;
        }
        
        .code-block-container .copy-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #444;
            border: none;
            color: #fff;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }
        
        .code-block-container .copy-btn:hover {
            background: #555;
        }
        
        #upload-pdf-btn:hover {
            color: #fff;
        }
        
        .pdf-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 5px 0;
            color: #aaa;
        }
        
        .pdf-item.active {
            color: #4CAF50;
        }
        
        .pdf-item input[type="checkbox"] {
            margin-left: 10px;
        }
        
        .delete-pdf-btn {
            background: rgba(255, 68, 68, 0.2);
            border: none;
            color: #ff4444;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 12px;
            transition: background 0.2s;
        }
        
        .delete-pdf-btn:hover {
            background: rgba(255, 68, 68, 0.4);
        }
        
        .settings-container {
            position: absolute;
            bottom: 70px;
            left: 16px;
            right: 16px;
        }
        
        .settings-btn {
            background: #232423;
            padding: 12px 16px;
            border: none;
            border-radius: 8px;
            color: #fff;
            width: 100%;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .settings-btn:hover {
            background: #2b2b2b;
        }
        
        .settings-dropdown {
            display: none;
            position: absolute;
            bottom: 50px;
            background: #2b2b2b;
            border: 1px solid #444;
            border-radius: 8px;
            width: 100%;
            z-index: 1000;
        }
        
        .settings-dropdown.show {
            display: block;
        }
        
        .dropdown-item {
            padding: 12px 16px;
            color: #fff;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .dropdown-item.username {
            cursor: default;
            color: #aaa;
            pointer-events: none;
        }
        
        .dropdown-item:not(.username):hover {
            background: #3b3b3b;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .input-box {
                width: calc(100% - 20px);
                max-width: none;
            }
            
            .chat-messages {
                margin-bottom: 100px;
            }
        }
        
        @media (max-width: 480px) {
            .input-box {
                width: calc(100% - 10px);
                bottom: 10px;
            }
            
            .chat-messages {
                margin-bottom: 120px;
            }
            
            textarea {
                padding: 8px 15px;
            }
            
            button {
                padding: 8px 16px;
            }
        }
        
        @media (min-width: 769px) {
            .chat-area {
                filter: none !important;
                pointer-events: auto !important;
            }
            .input-box.disabled {
                pointer-events: auto;
                opacity: 1;
            }
            .input-box.shifted {
                transform: translateX(-370px); /* Adjusted for sidebar */
            }
            .chat-messages.shifted {
                transform: translateX(65px);
            }
            #chat-header-title.shifted {
                margin-left: 260px;
            }
        }

        .profile-pic {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            object-fit: cover;
            flex-shrink: 0;
        }

        .message-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .image-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #333;
}

.image-item.active {
    color: #4CAF50;
}

.delete-image-btn {
    background: rgba(255, 68, 68, 0.2);
    border: none;
    color: #ff4444;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 14px;
    transition: background 0.2s;
}

.delete-image-btn:hover {
    background: rgba(255, 68, 68, 0.4);
}

/* Make bot message box completely transparent */
.bot-message .message-box {
    background: transparent !important;
    border: none !important;
    backdrop-filter: none !important;
}

.edit-convo-btn {
    position: absolute;
    right: 28px; /* Positioned to the left of delete button */
    top: 50%;
    transform: translateY(-50%);
    background: transparent;
    border: none;
    color: #aaa;
    cursor: pointer;
    font-size: 14px;
    opacity: 0;
    transition: opacity 0.2s;
}

.conversation-item:hover .edit-convo-btn {
    opacity: 1;
}

.edit-convo-btn:hover {
    color: #4d84ff;
}

.delete-convo-btn {
    position: absolute;
    right: 8px; /* Keep at right edge */
    top: 50%;
    transform: translateY(-50%);
    background: transparent;
    border: none;
    color: #ff4444;
    cursor: pointer;
    font-size: 14px;
    opacity: 0;
    transition: opacity 0.2s;
}

/* Add these styles for the drag-and-drop overlay */
.drag-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(77, 132, 255, 0.85); /* A vibrant but semi-transparent blue */
  border: 2px dashed #ffffff;
  border-radius: 14px; /* Match the input box's border-radius */
  display: none; /* Hidden by default */
  align-items: center;
  justify-content: center;
  pointer-events: none; /* Allows the 'drop' event to hit the underlying element */
  z-index: 20;
  transition: background .2s ease;
}

.drag-overlay-text {
  color: white;
  font-size: 1.2em;
  font-weight: 600;
  text-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* This class will be toggled by JavaScript to show the overlay */
.input-box.dragging .drag-overlay {
  display: flex;
}

#file-indicator-container {
  /* New positioning rules */
  position: fixed;
  bottom: 95px; /* A fallback initial position */
  left: 50%;
  transform: translateX(-50%);
  z-index: 14; /* Position it just below the input box (z-index 15) */
  transition: bottom 0.2s ease-out; /* Smooth transition when moving */

  /* Keep existing layout styles and add width constraints for alignment */
  width: 100%;
  max-width: 800px;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 0 20px;
  box-sizing: border-box;
  pointer-events: none; /* Allows clicks to go through to messages if no pills are hit */
}

/* Make individual pills clickable again */
.file-pill {
  pointer-events: auto;
  background-color: #3a3a3a;
  color: #e0e0e0;
  padding: 5px 10px;
  border-radius: 16px; /* This makes it pill-shaped */
  font-size: 13px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid #555;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  max-width: 200px; /* Prevent very long filenames from breaking layout */
}


/* Add these new rules to your <style> block */

.message-attachments {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px; /* Adds space between attachments and the message text */
}

.attachment-pill {
  background-color: #4a4a4a;
  color: #dcdcdc;
  padding: 4px 10px;
  border-radius: 14px;
  font-size: 13px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid #5f5f5f;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
}

.attachment-pill::before {
  content: '📄'; /* Default file icon */
  font-size: 14px;
}

.attachment-pill.type-image::before {
  content: '🖼️'; /* Image icon */
}

    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <button id="sidebar-toggle" onclick="toggleSidebar()">☰</button>
        <div class="sidebar-links">
            <a href="/create-bot">Create Bot</a>
            <a href="/bots">Browse Bots</a>
        </div>
        <div id="conversation-list"></div>
        <div class="settings-container">
            <button class="settings-btn">Settings</button>
            <div class="settings-dropdown">
                <div class="dropdown-item username">{{ username }}</div>
                <div class="dropdown-item" onclick="window.location.href='/user-profile'">User Profile</div>
                <div class="dropdown-item" onclick="window.location.href='/my-bots'">My Bots</div>
                <div class="dropdown-item" onclick="logout()">Log Out</div>
            </div>
        </div>
    </div>
    
    <div class="chat-area">
        <div class="chat-header">
            <button id="sidebar-toggle-header" onclick="toggleSidebar()">☰</button>
            <div id="chat-header-title">Omni.ai</div>
        </div>
        <div class="chat-messages" id="chat-messages"></div>
        
        <!-- ▼▼▼ ADD THIS NEW DIV ▼▼▼ -->
<div id="file-indicator-container"></div>
<!-- ▲▲▲ ADD THIS NEW DIV ▲▲▲ -->

       <div class="input-box" id="input-box">
        <div class="drag-overlay" id="drag-overlay">
    <div class="drag-overlay-text">Add your files</div>
        </div>

  <!-- + button -->
  <button id="add-files-btn" class="bar-btn" title="Add files">+</button>

  <textarea id="user-input" placeholder="Message…" rows="1"></textarea>

  <button id="send-stop-btn" class="bar-btn">➤</button>

  <!-- popup menu (unchanged) -->
  <div id="files-popup" class="files-popup">
    <button class="popup-header" id="upload-image-btn">🖼️ Images</button>
    <div id="image-popup" class="popup-content">
      <div id="image-list"></div>
      <input type="file" id="image-input" accept="image/*" style="display:none" onchange="uploadImage()">
      <button onclick="document.getElementById('image-input').click()" style="width:100%; margin-top:10px; background:#4CAF50; border:none; padding:6px; border-radius:4px;">Upload Image</button>
    </div>

    <button class="popup-header" id="upload-pdf-btn">📎 Files</button>
    <div id="pdf-popup" class="popup-content">
      <div id="pdf-list"></div>
      <input type="file" id="pdf-input" accept=".pdf" style="display:none" onchange="uploadPDF()">
      <button onclick="document.getElementById('pdf-input').click()" style="width:100%; margin-top:10px; background:#4CAF50; border:none; padding:6px; border-radius:4px;">Upload File</button>
    </div>
  </div>
</div>
    </div>
</div>

    <script>

    let uploadedFileIndices = [];

    /**
 * Truncates a long PDF filename using the "first4...last3.pdf" format.
 * If the filename (without extension) is 10 characters or less, it returns the original name.
 * @param {string} fullName The full filename, e.g., "verylongdocumentname.pdf"
 * @returns {string} The truncated or original filename.
 */
function truncateFilename(fullName) {
    // Find the last dot to separate the name from the extension
    const lastDotIndex = fullName.lastIndexOf('.');
    if (lastDotIndex === -1) {
        return fullName; // Return original if no extension is found
    }

    const name = fullName.substring(0, lastDotIndex);
    const extension = fullName.substring(lastDotIndex); // e.g., ".pdf"

    // Only truncate if the name part is longer than 10 characters
    if (name.length > 10) {
        const firstPart = name.substring(0, 4);
        const lastPart = name.substring(name.length - 3);
        return `${firstPart}...${lastPart}${extension}`;
    }

    // Otherwise, return the original full name
    return fullName;
}

    function createFileIndicatorPill(fileName, fileType, number) {
    const container = document.getElementById('file-indicator-container');

    // --- NEW: Decide which name to display ---
    // Apply truncation only if the file is a PDF.
    const displayName = fileType === 'pdf' ? truncateFilename(fileName) : fileName;

    const pill = document.createElement('div');
    pill.className = 'file-pill';
    pill.dataset.filename = fileName;
    pill.dataset.filetype = fileType;

    if (fileType === 'image' && number) {
        pill.dataset.number = number;
    }

    // Use displayName for the text but the original fileName for the hover title
    pill.innerHTML = `
        <span class="filename" title="${fileName}">${displayName}</span>
        <button class="remove-pill" title="Delete file">×</button>
    `;

    pill.querySelector('.remove-pill').addEventListener('click', (e) => {
        e.stopPropagation();
        if (fileType === 'pdf') {
            deletePDF(fileName, e);
        } else if (fileType === 'image') {
            deleteImage(fileName, e);
        }
    });

    if (fileType === 'image' && number) {
        pill.addEventListener('click', () => {
            alert(`This file's assigned number is: ${pill.dataset.number}`);
        });
    }

    container.appendChild(pill);
    updatePillContainerPosition();
}

    function updatePillContainerPosition() {
    const inputBox = document.getElementById('input-box');
    const pillContainer = document.getElementById('file-indicator-container');
    const messagesDiv = document.getElementById('chat-messages');

    if (!inputBox || !pillContainer || !messagesDiv) return;

    // Wait a fraction of a second for the DOM to update heights after a change
    setTimeout(() => {
        const inputBoxHeight = inputBox.offsetHeight;
        const pillContainerHeight = pillContainer.offsetHeight;

        // The input box is positioned 20px from the viewport bottom
        const baseBottomOffset = 20; 
        const gap = 10; // 10px of space between input box and pills

        // 1. Position the pill container dynamically
        // Its bottom position is the input box's height + its offset + a gap
        pillContainer.style.bottom = `${baseBottomOffset + inputBoxHeight + gap}px`;
        
        // 2. Adjust the message list's bottom margin to prevent overlap
        // The total space needed is the height of the input box and the pill container, plus offsets
        const requiredMargin = inputBoxHeight + pillContainerHeight + baseBottomOffset + gap;
        messagesDiv.style.marginBottom = `${requiredMargin}px`;
    }, 50); // A 50ms delay is usually sufficient
}

    // --- START: Drag and Drop File Upload ---

const inputBox = document.getElementById('input-box');

// Prevent default browser behaviors for drag-and-drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  inputBox.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Add a 'dragging' class to style the drop zone on drag enter/over
inputBox.addEventListener('dragenter', () => inputBox.classList.add('dragging'), false);
inputBox.addEventListener('dragover', () => inputBox.classList.add('dragging'), false);

// Remove the 'dragging' class on drag leave or drop
inputBox.addEventListener('dragleave', () => inputBox.classList.remove('dragging'), false);
inputBox.addEventListener('drop', () => inputBox.classList.remove('dragging'), false);

// Handle the actual file drop
inputBox.addEventListener('drop', handleDrop, false);

async function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  // Process all dropped files
  for (const file of files) {
      const fileName = file.name.toLowerCase();

      if (fileName.endsWith('.pdf')) {
          console.log('Uploading dropped PDF:', file.name);
          await uploadDroppedFile(file, '/upload_pdf', 'pdf', updatePDFList);
      } else if (file.type.startsWith('image/')) {
          console.log('Uploading dropped Image:', file.name);
          await uploadDroppedFile(file, '/upload_image', 'image', updateImageList);
      } else {
          console.warn('Unsupported file type dropped:', file.name);
          alert(`File type not supported: ${file.name}`);
      }
  }
}

async function uploadDroppedFile(file, endpoint, formDataKey, uiUpdateFunction) {
    const formData = new FormData();
    formData.append(formDataKey, file);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            await uiUpdateFunction();
            
            if (formDataKey === 'image') {
                alert(`Image uploaded. Assigned number: ${data.imageNumber}`);
                // --- Add image index to the list ---
                uploadedFileIndices.push(data.imageNumber);
                createFileIndicatorPill(data.filename, 'image', data.imageNumber);
                await toggleImage(data.filename, true);

            } else if (formDataKey === 'pdf') {
                alert(`PDF uploaded. Assigned number: ${data.pdfNumber}`);
                // --- Add PDF index and show the list alert ---
                uploadedFileIndices.push(data.pdfNumber);
                alert(uploadedFileIndices.join(','));
                createFileIndicatorPill(data.filename, 'pdf', data.pdfNumber);
                await togglePDF(data.filename, true);
            }

        } else {
            alert(`Failed to upload ${file.name}: ` + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error uploading dropped file:', error);
        alert(`A network error occurred while uploading ${file.name}.`);
    }
}

function createFileIndicatorPill(fileName, fileType, number) {
    const container = document.getElementById('file-indicator-container');

    // Create the pill element
    const pill = document.createElement('div');
    pill.className = 'file-pill';
    pill.dataset.filename = fileName; // Store the name for removal logic

    // If it's an image and a number is provided, store it in a data attribute
    if (fileType === 'image' && number) {
        pill.dataset.number = number;
    }

    pill.innerHTML = `
        <span class="filename">${fileName}</span>
        <button class="remove-pill" title="Delete file">×</button>
    `;

    // --- MODIFIED SECTION ---
    // The 'X' button now directly calls the appropriate DELETE function
    pill.querySelector('.remove-pill').addEventListener('click', (e) => {
        e.stopPropagation();
        
        if (fileType === 'pdf') {
            // Call the main delete function for PDFs
            deletePDF(fileName, e);
        } else if (fileType === 'image') {
            // Call the main delete function for images
            deleteImage(fileName, e);
        }
    });
    // --- END OF MODIFICATION ---

    // If it's an image with a number, add a click listener to the whole pill
    if (fileType === 'image' && number) {
        pill.addEventListener('click', () => {
            // Show an alert with the file's assigned number
            alert(`This file's assigned number is: ${pill.dataset.number}`);
        });
    }

    container.appendChild(pill);
    updatePillContainerPosition(); // Adjust positions after adding a pill
}


    const lockedBranchIds = new Map();

    // Global, once-only listener for the send/stop button
document.getElementById('send-stop-btn').addEventListener('click', toggleSendStop);

    document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        toggleSendStop();
    }
});

    document.getElementById('add-files-btn').addEventListener('click', e => {
  e.stopPropagation();
  const p = document.getElementById('files-popup');
  p.style.display = p.style.display === 'flex' ? 'none' : 'flex';
});
document.addEventListener('click', e => {
  if (!e.target.closest('#add-files-btn') && !e.target.closest('#files-popup')) {
    document.getElementById('files-popup').style.display = 'none';
  }
});


    // New functions for image management
    document.getElementById('upload-image-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        const popup = document.getElementById('image-popup');
        popup.style.display = popup.style.display === 'none' ? 'block' : 'none';
        updateImageList();
    });

       // Add this logic to the uploadImage function
async function uploadImage() {
    const fileInput = document.getElementById('image-input');
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            alert(`Image uploaded. Assigned number: ${data.imageNumber}`);
            
            // --- Add the index to the global list ---
            uploadedFileIndices.push(`${data.imageNumber}`);
            
            createFileIndicatorPill(data.filename, 'image', data.imageNumber);
            await toggleImage(data.filename, true); 
            document.getElementById('image-popup').style.display = 'block';
        } else {
            alert('Failed to upload image: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to upload image');
    }
    fileInput.value = '';
}


    async function updateImageList() {
    const response = await fetch('/get_images');
    const data = await response.json();
    const imageList = document.getElementById('image-list');
    imageList.innerHTML = '';

    data.images.forEach(image => {
        const isSent = image.sent || false;
        const div = document.createElement('div');
        div.className = `image-item ${image.active ? 'active' : ''} ${isSent ? 'sent-item' : ''}`;
        
        // Display index with 'N' if sent
        const displayName = `${image.name} (${image.number}${isSent ? 'N' : ''})`;

        div.innerHTML = `
            <span>${displayName}</span>
            <div style="display: flex; align-items: center;">
                <input type="checkbox" ${image.active ? 'checked' : ''} onchange="toggleImage('${image.name}', this.checked)" ${isSent ? 'disabled' : ''}>
                <button class="delete-image-btn" onclick="deleteImage('${image.name}', event)">×</button>
            </div>
        `;
        imageList.appendChild(div);
    });
}

    async function toggleImage(imageName, active) {
    // This function is called when a checkbox is clicked
    try {
        const response = await fetch('/toggle_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_name: imageName, active: active })
        });
        const data = await response.json();

        if (data.success) {
            updateImageList(); // Refresh the list
        } else {
            alert('Failed to update image status: ' + (data.error || 'Unknown error'));
            // Revert the checkbox visually if the backend failed
            const checkbox = document.querySelector(`input[onchange="toggleImage('${imageName}', this.checked)"]`);
            if(checkbox) checkbox.checked = !active;
        }
    } catch (error) {
        alert('A network error occurred while updating image status.');
    }
}

    async function deleteImage(imageName, event) {
    event.stopPropagation();
    if (!confirm(`Are you sure you want to delete "${imageName}"?`)) return;

    try {
        const response = await fetch('/delete_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_name: imageName })
        });
        const data = await response.json();
        if (data.success) {
            // This updates the dropdown list in the sidebar
            updateImageList();

            // --- ADD THIS BLOCK TO REMOVE THE PILL ---
            const pillToRemove = document.querySelector(`.file-pill[data-filename="${imageName}"]`);
            if (pillToRemove) {
                pillToRemove.remove();

                updatePillContainerPosition(); // Adjust positions after removing a pill
            }
            // --- END OF NEW BLOCK ---

        } else {
            alert('Failed to delete image: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Failed to delete image');
    }
}

    // Update click handler to include image popup
    document.addEventListener('click', (event) => {
        const pdfPopup = document.getElementById('pdf-popup');
        const pdfBtn = document.getElementById('upload-pdf-btn');
        const imagePopup = document.getElementById('image-popup');
        const imageBtn = document.getElementById('upload-image-btn');
        
        if (!pdfPopup.contains(event.target) && event.target !== pdfBtn) {
            pdfPopup.style.display = 'none';
        }
        if (!imagePopup.contains(event.target) && event.target !== imageBtn) {
            imagePopup.style.display = 'none';
        }
    });

        function adjustTextareaHeight() {
        const textarea = document.querySelector('textarea');
        textarea.style.height = 'auto'; // Reset height
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'; // Set height up to max-height

        updatePillContainerPosition(); // Adjust pill container position on height change
    }

    document.querySelector('textarea').addEventListener('input', adjustTextareaHeight);

        document.querySelector('.settings-btn').addEventListener('click', function(e) {
            e.stopPropagation();
            document.querySelector('.settings-dropdown').classList.toggle('show');
        });

        document.addEventListener('click', function(e) {
            if (!e.target.closest('.settings-container')) {
                document.querySelector('.settings-dropdown').classList.remove('show');
            }
        });

        let currentBotName = "Omni.ai";
        let currentBotPhoto = "";
        let currentUserPhoto = '';
        let isStreaming = false;
        let currentStreamController = null;

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const inputBox = document.getElementById('input-box');
            const chatHeaderTitle = document.getElementById('chat-header-title');
            const chatMessages = document.getElementById('chat-messages');
            const isOpen = sidebar.classList.toggle('open');
            
            if (window.innerWidth <= 768) {
                inputBox.classList.toggle('disabled', isOpen);
                if (isOpen) {
                    document.getElementById('user-input').blur();
                }
            } else {
                chatHeaderTitle.classList.toggle('shifted', isOpen);
                inputBox.classList.toggle('shifted', isOpen);
                chatMessages.classList.toggle('shifted', isOpen);
            }
        }

        function toggleSendStop() {
            if (isStreaming) {
                stopMessage();
            } else {
                sendMessage();
            }
        }



// 1) only replace the spinner inside a .bot-message
function replaceLoadingSpinnerWithMessage(text) {
  // look *only* inside the bot‑bubble
  const spinner = document.querySelector('.bot-message .loading-spinner');
  if (!spinner) return;

  // replace just that spinner+text container
  const container = spinner.parentElement;
  container.innerHTML = text;
}

// Inside your <script> tag (message-41.txt)

// Replace the existing stopMessage function with this improved version:
async function stopMessage() {
    if (!currentStreamController) return; // [cite: 238]

    const sendStopBtn = document.getElementById('send-stop-btn'); // [cite: 242]

    try {
        // 1. Call the backend to signal stop and get the correct index/history
        const resp = await fetch('/stop_stream', { // [cite: 239]
            method: 'POST', // [cite: 239]
            headers: { 'Content-Type': 'application/json' } // [cite: 239]
        });
        const data = await resp.json(); // [cite: 240]

        if (!data.success) { // [cite: 240]
            console.error('Stop failed:', data.error); // [cite: 240]
            // Still attempt to abort client-side stream
            if (currentStreamController) currentStreamController.abort(); // [cite: 241]
            isStreaming = false; // [cite: 241]
            sendStopBtn.innerHTML = '&#10148;'; // [cite: 242]
            currentStreamController = null; // [cite: 254]
            return;
        }

        // 2. Abort the client-side fetch request
        currentStreamController.abort(); // [cite: 241]
        isStreaming = false; // [cite: 241]
        sendStopBtn.innerHTML = '&#10148;'; // [cite: 242]

        // 3. Find the *last* bot message div in the DOM (the one with the spinner)
        const messagesDiv = document.getElementById('chat-messages'); // [cite: 242]
        const lastBotMessageDiv = messagesDiv.querySelector('.bot-message:last-of-type'); // [cite: 247]

        if (lastBotMessageDiv) {
            // 4. Use the index provided by the server
            const serverIndex = data.stopped_message_index; // Use the index from the server response
            lastBotMessageDiv.dataset.index = serverIndex; // Update the DOM element's index [cite: 248]

            // 5. Get the final text for this message from the server's history snapshot
            const finalServerText = data.history && data.history[serverIndex] ? data.history[serverIndex].text : ''; // [cite: 249]

            // 6. Update the message content, remove spinner, and potentially show controls
            const messageContentDiv = lastBotMessageDiv.querySelector('.message-content div:not(.message-controls)'); // [cite: 250]
            if (messageContentDiv) {
                messageContentDiv.innerHTML = formatMessageText(finalServerText); // Update with final text (might be empty) [cite: 250]
            }

            const controls = lastBotMessageDiv.querySelector('.message-controls'); // [cite: 250]
            if (controls) {
                 // Ensure controls are correctly linked to the serverIndex
                 const regenBtn = controls.querySelector('.regenerate-btn');
                 const deleteBtn = controls.querySelector('.delete-message-btn');
                 const branchBtn = controls.querySelector('.branch-btn');

                 if (regenBtn) regenBtn.onclick = () => regenerateMessage(serverIndex); // [cite: 252]
                 if (deleteBtn) deleteBtn.onclick = () => deleteMessage(serverIndex); // [cite: 252]
                 if (branchBtn) {
                     branchBtn.onclick = (event) => toggleBranchDropdown(serverIndex, event); // [cite: 252]
                     const branchMenu = controls.querySelector('.branch-menu');
                     if(branchMenu) branchMenu.id = `branch-menu-${serverIndex}`; // [cite: 253]
                 }

                 // Show controls now that generation is stopped
                 controls.style.display = 'flex'; // [cite: 251]
                 controls.querySelectorAll('button').forEach(btn => btn.style.display = ''); // [cite: 252]
                 await updateBranchDropdown(serverIndex); // Update branches for this index [cite: 253]
            }
             messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
        }

    } catch (error) { // [cite: 253]
        console.error('Error stopping stream:', error); // [cite: 253]
        // Ensure cleanup happens even on error
        if (currentStreamController && !currentStreamController.signal.aborted) {
            currentStreamController.abort();
        }
        isStreaming = false;
        sendStopBtn.innerHTML = '&#10148;';
    } finally {
        currentStreamController = null; // [cite: 254]
    }
}





        document.getElementById('upload-pdf-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            const popup = document.getElementById('pdf-popup');
            popup.style.display = popup.style.display === 'none' ? 'block' : 'none';
            updatePDFList();
        });

        // Modify the uploadPDF function like this
async function uploadPDF() {
    const fileInput = document.getElementById('pdf-input');
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('pdf', file);

    try {
        const response = await fetch('/upload_pdf', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            alert(`PDF uploaded. Assigned number: ${data.pdfNumber}`);

            // --- Add index to list and show the new alert ---
            uploadedFileIndices.push(data.pdfNumber);
            alert(uploadedFileIndices.join(','));
            // --- End of new logic ---
            
            createFileIndicatorPill(data.filename, 'pdf', data.pdfNumber);
            await togglePDF(data.filename, true);
            document.getElementById('pdf-popup').style.display = 'block';
        } else {
            alert('Failed to upload PDF: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to upload PDF');
    }
    fileInput.value = '';
}

        async function updatePDFList() {
    const response = await fetch('/get_pdfs');
    const data = await response.json();
    const pdfList = document.getElementById('pdf-list');
    pdfList.innerHTML = '';

    data.pdfs.forEach(pdf => {
        const isSent = pdf.sent || false;
        const displayName = truncateFilename(pdf.name);
        // Display index with 'N' if sent
        const fullDisplayName = `${displayName} (${pdf.number}${isSent ? 'N' : ''})`;

        const div = document.createElement('div');
        div.className = `pdf-item ${pdf.active ? 'active' : ''} ${isSent ? 'sent-item' : ''}`;
        
        div.innerHTML = `
            <span title="${pdf.name}">${fullDisplayName}</span>
            <div style="display: flex; align-items: center;">
                <input type="checkbox" ${pdf.active ? 'checked' : ''} onchange="togglePDF('${pdf.name}', this.checked)" ${isSent ? 'disabled' : ''}>
                <button class="delete-pdf-btn" onclick="deletePDF('${pdf.name}', event)">×</button>
            </div>
        `;
        pdfList.appendChild(div);
    });
}

        async function deletePDF(pdfName, event) {
    event.stopPropagation();
    if (!confirm(`Are you sure you want to delete "${pdfName}"?`)) return;

    try {
        const response = await fetch('/delete_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pdf_name: pdfName })
        });
        const data = await response.json();
        if (data.success) {
            // This updates the dropdown list in the sidebar
            updatePDFList();

            // --- ADD THIS BLOCK TO REMOVE THE PILL ---
            const pillToRemove = document.querySelector(`.file-pill[data-filename="${pdfName}"]`);
            if (pillToRemove) {
                pillToRemove.remove();

                updatePillContainerPosition(); // Adjust positions after removing a pill
            }
            // --- END OF NEW BLOCK ---
            
        } else {
            alert('Failed to delete PDF: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Failed to delete PDF');
    }
}

        // Do the same for togglePDF
async function togglePDF(pdfName, active) {
    try {
        const response = await fetch('/toggle_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pdf_name: pdfName, active: active })
        });
        const data = await response.json();
        
        if (data.success) {
            updatePDFList(); // Refresh the list
        } else {
            alert('Failed to update PDF status: ' + (data.error || 'Unknown error'));
            const checkbox = document.querySelector(`input[onchange="togglePDF('${pdfName}', this.checked)"]`);
            if(checkbox) checkbox.checked = !active;
        }
    } catch (error) {
        alert('A network error occurred while updating PDF status.');
    }
}

        document.addEventListener('click', (event) => {
            const popup = document.getElementById('pdf-popup');
            const btn = document.getElementById('upload-pdf-btn');
            if (!popup.contains(event.target) && event.target !== btn) {
                popup.style.display = 'none';
            }
        });

        const textarea = document.getElementById('user-input');
        
        function autoResize() {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        }
        
        function handleInput(event) {
            autoResize();
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        textarea.addEventListener('input', autoResize);

        function formatMessageText(text) {
            text = text.replace(/\*\*(.*?)\*\*/g, '<em>$1</em>');
            text = text.replace(/```([\s\S]*?)```/g, (match, code) => {
                return `<div class="code-block-container"><pre><code>${code.trim()}</code></pre><button class="copy-btn" onclick="copyCode(this)">Copy</button></div>`;
            });
            return text;
        }

        function copyCode(button) {
            const codeBlock = button.previousElementSibling.querySelector('code');
            const codeText = codeBlock.textContent;
            navigator.clipboard.writeText(codeText).then(() => {
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                button.style.background = '#4CAF50';
                setTimeout(() => {
                    button.textContent = originalText;
                    button.style.background = '#444';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy code:', err);
            });
        }

function displayMessage(msg, isUser, index) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.dataset.index = index;
    
    const photoSrc = msg.profile_photo || (isUser ? currentUserPhoto : currentBotPhoto) || 'https://via.placeholder.com/32';
    const senderName = isUser ? '{{ username }}' : currentBotName;

    // --- REFACTORED LOGIC ---
    // Read attachments directly from the message object. Default to an empty array.
    const attachments = msg.attachments || [];
    let attachmentsHTML = '';
    if (attachments.length > 0) {
        attachmentsHTML += '<div class="message-attachments">';
        attachments.forEach(att => {
            const displayName = att.type === 'pdf' ? truncateFilename(att.filename) : att.filename;
            const typeClass = att.type === 'image' ? 'type-image' : '';
            attachmentsHTML += `<div class="attachment-pill ${typeClass}" title="${att.filename}">${displayName}</div>`;
        });
        attachmentsHTML += '</div>';
    }
    // --- END ---

    // The rest of the function's innerHTML remains the same
    messageDiv.innerHTML = `
        <img src="${photoSrc}" class="profile-pic" alt="${senderName}">
        <div class="message-content">
            <div class="message-box">
                <strong>${senderName}:</strong>
                ${attachmentsHTML} <!-- This now renders saved attachments -->
                <div>${formatMessageText(msg.text)}</div>
                <div class="message-controls">
                    ${!isUser ? `
                        <button class="regenerate-btn" title="Regenerate" onclick="regenerateMessage(${index})">↻</button>
                    ` : `
                        <button class="edit-message-btn" title="Edit message" onclick="editMessage(${index})">✎</button>
                    `}
                    <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${index})">×</button>
                    <div style="position: relative; display: inline-block;">
                        <button class="branch-btn" onclick="toggleBranchDropdown(${index}, event)">…</button>
                        <div class="branch-menu" id="branch-menu-${index}" style="display: none;"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    updateBranchDropdown(index);
} 

// MODIFIED: This version correctly performs both actions for user messages.
async function toggleBranchDropdown(messageIndex, event) {
    event.preventDefault();
    event.stopPropagation(); // Prevent the click from closing the menu immediately.

    // Find the message element in the DOM to check its type.
    const msgEl = document.querySelector(`.message[data-index="${messageIndex}"]`);
    if (!msgEl) return;

    // --- TEST FEATURE FOR USER MESSAGES (RUNS FIRST) ---
    if (msgEl.classList.contains('user-message')) {
        // As requested, do nothing for the very first message.
        if (messageIndex > 0) {
            try {
                const response = await fetch('/history');
                const data = await response.json();
                const history = data.history;

                const precedingMessage = history[messageIndex - 1];
                if (precedingMessage && precedingMessage.sender === 'bot') {
                    const uniqueId = precedingMessage.unique_id;
                    if (uniqueId) {
                        alert(`ID of preceding bot message: ${uniqueId}`);
                    } else {
                        alert("The preceding bot message does not have a unique ID stored.");
                    }
                } else {
                     alert("Could not find a preceding bot message.");
                }
            } catch (error) {
                console.error("Error fetching history for ID test:", error);
                alert("An error occurred while fetching the message ID.");
            }
        }
        // IMPORTANT: We do NOT 'return' here. We let the code flow to the dropdown logic.
    }
    
    // --- DROPDOWN LOGIC (RUNS FOR BOTH USER AND BOT MESSAGES) ---
    const menu = document.getElementById(`branch-menu-${messageIndex}`);
    if (!menu) {
        // If the menu element doesn't exist for some reason, we can't proceed.
        return;
    }

    const parentMessage = menu.closest('.message');
    const isCurrentlyVisible = menu.style.display === 'block';

    // First, hide all other menus on the page.
    document.querySelectorAll('.branch-menu').forEach(m => m.style.display = 'none');
    document.querySelectorAll('.message.menu-active').forEach(m => m.classList.remove('menu-active'));

    // If this specific menu wasn't already visible, populate and show it.
    if (!isCurrentlyVisible) {
        // This function populates the dropdown with the correct branches.
        // It's designed to work for both user (edit) and bot (regen) branches.
        await updateBranchDropdown(messageIndex); 
        
        menu.style.display = 'block';
        if (parentMessage) {
            parentMessage.classList.add('menu-active');
        }
    }
}

function filterBranchesByType(branches, index, isBotMessage) {
    return Object.entries(branches).filter(([branchId, branchInfo]) => {
        // Always include main branch
        if (branchId === 'main') return true;
        
        // Only show branches that originated from this exact message index
        const isThisMessageBranch = branchInfo.action_message_index === index;
        
        // Match branch type to message type
        const isCorrectType = (isBotMessage && branchId.startsWith('B')) || 
                            (!isBotMessage && branchId.startsWith('U'));
        
        return isThisMessageBranch && isCorrectType;
    });
}

async function updateBranchDropdown(messageIndex) {
    const branchMenu = document.getElementById(`branch-menu-${messageIndex}`);
    
    if (!branchMenu) return;
    
    // Start with a truly blank dropdown.
    branchMenu.innerHTML = '';
    
    const { branches, active_branch } = await (await fetch('/branches')).json();
    
    const options = new Map();
    
    // --- CORRECTED GATEKEEPER LOGIC ---
    // Check if this message index is actually the start of a valid branch
    const isValidBranchStart = Object.values(branches).some(
        branchInfo => 
            branchInfo.action_message_index === messageIndex && 
            branchInfo.parent_message_index !== undefined
    );
    
    // If this message isn't a valid branch start point, don't show dropdown
    if (!isValidBranchStart) {
        return;
    }
    
    // Additional check: For non-main branches, verify the parent relationship
    const hasValidParentBranch = Object.values(branches).some(branchInfo => {
        if (branchInfo.action_message_index !== messageIndex) return false;
        if (branchInfo.parent_branch === 'main' || !branchInfo.parent_branch) return true;
        return branches[branchInfo.parent_branch] !== undefined;
    });
    
    if (!hasValidParentBranch) {
        return;
    }
    // --- END OF CORRECTED GATEKEEPER LOGIC ---
    
    const msgEl = document.querySelector(`.message[data-index="${messageIndex}"]`);
    if (!msgEl) return;
    
    // The logic differs for user messages and bot messages.
    if (msgEl.classList.contains('user-message')) {
        // --- LOGIC FOR USER MESSAGES ---
        for (const [branchId, branchInfo] of Object.entries(branches)) {
            if (branchInfo.action_message_index === messageIndex && branchId.startsWith('U')) {
                options.set(branchId, branchId);
                
                if (branchInfo.parent_branch) {
                    options.set(branchInfo.parent_branch, '1 (Original)');
                }
            }
        }
    } else {
        // --- LOGIC FOR BOT MESSAGES ---
        const getTimelineRoot = (branchId) => {
            if (!branchId || branchId === 'main') return 'main';
            const baseId = branchId.split('-')[0];
            return baseId.startsWith('B') ? 'main' : baseId;
        };
        
        const activeTimelineRoot = getTimelineRoot(active_branch);
        options.set(activeTimelineRoot, '1 (Original)');
        
        for (const [branchId, branchInfo] of Object.entries(branches)) {
            if (branchInfo.action_message_index === messageIndex) {
                const versionTimelineRoot = getTimelineRoot(branchId);
                if (versionTimelineRoot === activeTimelineRoot) {
                    options.set(branchId, branchId);
                }
            }
        }
    }
    
    // --- COMMON RENDERING LOGIC ---
    if (options.size <= 1) {
        return;
    }
    
    const sortedOptions = new Map([...options.entries()].sort((a, b) => {
        if (a[1].includes('Original')) return -1;
        if (b[1].includes('Original')) return 1;
        return a[0].localeCompare(b[0], undefined, { numeric: true });
    }));
    
    for (const [branchId, displayText] of sortedOptions.entries()) {
        const item = document.createElement('div');
        item.className = 'branch-item';
        // Use the branch ID or a user-friendly name as the text content
        item.textContent = displayText; 
        item.dataset.branchId = branchId;
        
        if (branchId === active_branch) {
            item.classList.add('active');
        }
        
        item.onclick = () => switchBranch(branchId);
        branchMenu.appendChild(item);
    }
            // ▼▼▼ PASTE THIS ENTIRE BLOCK TO REPLACE THE OLD ONE AT THE END OF updateBranchDropdown ▼▼▼

    // --- START: Block to append and LOCK preceding bot message ID using sessionStorage ---
    const { conversation_id: convIdForLocking } = await (await fetch('/branches')).json(); // Get the conv_id
    
    // Only proceed if we are in an active conversation
    if (convIdForLocking) {
        const fullConversationData = (await (await fetch('/conversations')).json()).conversations[convIdForLocking];
        if (!fullConversationData) return;
        const allBranchesInConvo = fullConversationData.branches;

        const isUserMessage = msgEl.classList.contains('user-message');
        if (isUserMessage && messageIndex > 0) {
            
            branchMenu.querySelectorAll('.branch-item').forEach(item => {
                const branchId = item.dataset.branchId;
                // Create a unique, permanent key for sessionStorage
                const key = `${convIdForLocking}-${messageIndex}-${branchId}`;
                
                // 1. CHECK FOR A LOCKED ID FIRST
                const lockedId = sessionStorage.getItem(key);
                
                if (lockedId) {
                    // ID is already locked in storage. Use it.
                    const suffix = ` -(${lockedId})`;
                    if (!item.textContent.includes(suffix)) {
                        item.textContent += suffix;
                    }
                } else {
                    // 2. NO LOCKED ID FOUND. FIND IT, USE IT, AND LOCK IT.
                    let idToLock = null;
                    const branchData = allBranchesInConvo[branchId];
                    
                    if (branchData) {
                        // The source of truth is the branch's own historical data.
                        const historyForBranch = branchData.history;
                        const precedingBotMessage = historyForBranch[messageIndex - 1];

                        if (precedingBotMessage && precedingBotMessage.sender === 'bot' && precedingBotMessage.unique_id) {
                            idToLock = precedingBotMessage.unique_id;
                        }
                    }

                    if (idToLock) {
                        // Use the ID and then lock it into sessionStorage permanently for this session.
                        sessionStorage.setItem(key, idToLock); // <-- THE LOCK-IN STEP
                        const suffix = ` -(${idToLock})`;
                        if (!item.textContent.includes(suffix)) {
                            item.textContent += suffix;
                        }
                    }
                }
            });
        }
    }
    // --- END: Block to append and LOCK preceding bot message ID ---
// ▼▼▼ REPLACE the previous dynamic filtering block with THIS ONE at the end of updateBranchDropdown ▼▼▼

// --- START: Block for dynamic filtering AND contextual re-numbering of dropdown items ---
(async () => {
    try {
        const isUserMessage = msgEl.classList.contains('user-message');
        if (!isUserMessage || messageIndex === 0) return;

        // 1. Get the LIVE unique_id from the bot message right before this one.
        const historyResponse = await fetch('/history');
        if (!historyResponse.ok) return;
        const historyData = await historyResponse.json();
        const precedingMessage = historyData.history[messageIndex - 1];

        const activePrecedingBotId = (precedingMessage && precedingMessage.sender === 'bot' && precedingMessage.unique_id)
            ? precedingMessage.unique_id
            : null;

        if (!activePrecedingBotId) {
            branchMenu.querySelectorAll('.branch-item').forEach(item => {
                item.textContent = item.textContent.replace(/\s-\d+$/, ''); // Clean old numbers
                item.style.display = 'block';
            });
            return;
        }

        const idExtractorRegex = /\s-\((\d{6})\)$/;
        const visibleItems = []; // We will collect the items to be shown here

        // 2. First pass: Clean up old sequence numbers and determine visibility.
        branchMenu.querySelectorAll('.branch-item').forEach(item => {
            // ALWAYS clean up old sequence numbers first, e.g., "U1 -(123456)-1" -> "U1 -(123456)"
            item.textContent = item.textContent.replace(/\s-\d+$/, '');

            const match = item.textContent.match(idExtractorRegex);
            const itemBotId = match ? match[1] : null;

            if (itemBotId) {
                // This item has a stamped ID. It's a candidate for being visible.
                const isVisible = (itemBotId === activePrecedingBotId);
                item.style.display = isVisible ? 'block' : 'none';
                if (isVisible) {
                    visibleItems.push(item); // Add to our list to be numbered
                }
            } else {
                // This item does NOT have a stamped ID. Always show it and add to list.
                item.style.display = 'block';
                visibleItems.push(item);
            }
        });

        // 3. Second pass: Apply new, sequential counters ONLY to the visible items.
        visibleItems.forEach((item, index) => {
            const counter = index + 1; // Counter is 1-based (1, 2, 3...)
            item.textContent += ` -${counter}`;
        });

    } catch (error) {
        console.error("Error during dynamic branch filtering and numbering:", error);
    }
})();
// --- END: Block for dynamic filtering AND contextual re-numbering ---
}



        async function regenerateMessage(index) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = messagesDiv.children[index];
    if (!messageDiv || !messageDiv.classList.contains('bot-message')) return;

    // Clear messages after the one being regenerated
    const historyResponse = await fetch('/history');
    const history = await historyResponse.json();
    messagesDiv.innerHTML = '';
    history.history.slice(0, index).forEach((msg, idx) => {
        displayMessage(msg, msg.sender === 'user', idx);
    });

    // Display the bot message being regenerated
    messageDiv.className = 'message bot-message';
    messageDiv.dataset.index = index;
    messageDiv.innerHTML = `
        <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
        <div class="message-content">
            <strong>${currentBotName}:</strong>
            <div><div class="loading-spinner"></div> Regenerating...</div>
            <div class="message-controls" style="display: none;">
                <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${index})">↻</button>
                <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${index})">×</button>
                <div style="position: relative; display: inline-block;">
                    <button class="branch-btn" onclick="toggleBranchDropdown(${index}, event)">…</button>
                    <div class="branch-menu" id="branch-menu-${index}" style="display: none;"></div>
                </div>
            </div>
        </div>
    `;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    isStreaming = true;
    document.getElementById('send-stop-btn').innerHTML = '□';

    try {
        const controller = new AbortController();
        currentStreamController = controller;
        const response = await fetch('/regenerate_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message_index: index }),
            signal: controller.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let botText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done || !isStreaming) break;
            const chunk = decoder.decode(value, { stream: true });
            botText += chunk;
            messageDiv.innerHTML = `
                <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
                <div class="message-content">
                    <strong>${currentBotName}:</strong>
                    <div>${formatMessageText(botText)}</div>
                    <div class="message-controls" style="display: ${botText ? 'flex' : 'none'};">
                        <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${index})">↻</button>
                        <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${index})">×</button>
                        <div style="position: relative; display: inline-block;">
                            <button class="branch-btn" onclick="toggleBranchDropdown(${index}, event)">…</button>
                            <div class="branch-menu" id="branch-menu-${index}" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            `;
           
        }

        if (isStreaming) {
            messageDiv.querySelector('.message-controls').style.display = 'flex';
            await updateBranchDropdown(index);
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            messageDiv.innerHTML = `
                <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
                <div class="message-content">
                    <strong>${currentBotName}:</strong>
                    <div>Error: ${error.message}</div>
                    <div class="message-controls" style="display: flex;">
                        <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${index})">↻</button>
                        <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${index})">×</button>
                        <div style="position: relative; display: inline-block;">
                            <button class="branch-btn" onclick="toggleBranchDropdown(${index}, event)">…</button>
                            <div class="branch-menu" id="branch-menu-${index}" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            `;
            await updateBranchDropdown(index);
        }
    } finally {
        isStreaming = false;
        document.getElementById('send-stop-btn').innerHTML = '&#10148;';
        currentStreamController = null;
    }
}

        async function editMessage(index) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageElements = messagesDiv.querySelectorAll('.message');
            let targetElement = null;

            for (const element of messageElements) {
                if (parseInt(element.dataset.index) === index) {
                    targetElement = element;
                    break;
                }
            }

            if (!targetElement) return;

            const historyResponse = await fetch('/history');
            const history = await historyResponse.json();
            const message = history.history[index];

            if (message.sender !== 'user') {
                alert('You can only edit your own messages.');
                return;
            }

            const originalContent = targetElement.innerHTML;

            const textarea = document.createElement('textarea');
            textarea.value = message.text;
            textarea.style.width = '100%';
            textarea.style.padding = '8px';
            textarea.style.border = '1px solid #444';
            textarea.style.borderRadius = '4px';
            textarea.style.background = '#222';
            textarea.style.color = '#fff';
            textarea.style.fontSize = '15px';
            textarea.style.resize = 'none';
            textarea.style.overflowY = 'hidden';
            textarea.style.minHeight = '44px';
            textarea.style.lineHeight = '1.5';
            textarea.style.boxSizing = 'border-box';

            function autoResize() {
                textarea.style.height = 'auto';
                textarea.style.height = textarea.scrollHeight + 'px';
            }
            textarea.addEventListener('input', autoResize);
            autoResize();

            const buttonContainer = document.createElement('div');
            buttonContainer.style.marginTop = '8px';
            buttonContainer.style.display = 'flex';
            buttonContainer.style.gap = '8px';
            buttonContainer.style.justifyContent = 'flex-end';

            const cancelButton = document.createElement('button');
            cancelButton.textContent = 'Cancel';
            cancelButton.style.padding = '6px 12px';
            cancelButton.style.background = '#ff4444';
            cancelButton.style.border = 'none';
            cancelButton.style.borderRadius = '4px';
            cancelButton.style.color = '#fff';
            cancelButton.style.cursor = 'pointer';

            const okButton = document.createElement('button');
            okButton.textContent = 'OK';
            okButton.style.padding = '6px 12px';
            okButton.style.background = '#4CAF50';
            okButton.style.border = 'none';
            okButton.style.borderRadius = '4px';
            okButton.style.color = '#fff';
            okButton.style.cursor = 'pointer';

            targetElement.innerHTML = '';
            targetElement.appendChild(textarea);
            targetElement.appendChild(buttonContainer);
            buttonContainer.appendChild(cancelButton);
            buttonContainer.appendChild(okButton);

            textarea.focus();

            cancelButton.onclick = () => {
                targetElement.innerHTML = originalContent;
                updateBranchDropdown(index);
            };

            okButton.onclick = async () => {
                const newText = textarea.value.trim();
                if (newText === message.text.trim()) {
                    targetElement.innerHTML = originalContent;
                    updateBranchDropdown(index);
                    return;
                }

                if (!newText) {
                    alert('Message cannot be empty');
                    return;
                }

                okButton.disabled = true;
                okButton.style.background = '#2a6030';
                okButton.style.cursor = 'not-allowed';

                messagesDiv.innerHTML = '';
                const truncatedHistory = history.history.slice(0, index + 1);
                truncatedHistory.forEach((msg, idx) => {
                    if (idx < index) {
                        displayMessage(msg, msg.sender === 'user', idx);
                    }
                });

                displayMessage({ text: newText, profile_photo: message.profile_photo }, true, index);

                const botIndex = index + 1;
                const botMessageDiv = document.createElement('div');
                botMessageDiv.className = 'message bot-message';
                botMessageDiv.dataset.index = botIndex;
                botMessageDiv.innerHTML = `
                    <img src="${currentBotPhoto || 'https://via.placeholder.com/32'}" class="profile-pic" alt="${currentBotName}">
                    <div class="message-content">
                        <strong>${currentBotName}:</strong>
                        <div><div class="loading-spinner"></div> Generating Message...</div>
                        <div class="message-controls">
                            <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${botIndex})" style="display: none;">↻</button>
                            <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${botIndex})" style="display: none;">×</button>
                            <div style="position: relative; display: inline-block;">
                                <button class="branch-btn" onclick="toggleBranchDropdown(${botIndex}, event)" style="display: none;">…</button>
                                <div class="branch-menu" id="branch-menu-${botIndex}" style="display: none;"></div>
                            </div>
                        </div>
                    </div>
                `;
                messagesDiv.appendChild(botMessageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;

                isStreaming = true;
                document.getElementById('send-stop-btn').innerHTML = '□';

                try {
                    const controller = new AbortController();
                    currentStreamController = controller;
                    const response = await fetch('/edit_message', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            message_index: index,
                            new_text: newText 
                        }),
                        signal: controller.signal
                    });

                    if (!response.ok) {
                        const data = await response.json();
                        throw new Error(data.error || 'Failed to edit message');
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let botText = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done || !isStreaming) break;
                        const chunk = decoder.decode(value, { stream: true });
                        botText += chunk;
                        botMessageDiv.innerHTML = `
                            <img src="${currentBotPhoto || 'https://via.placeholder.com/32'}" class="profile-pic" alt="${currentBotName}">
                            <div class="message-content">
                                <strong>${currentBotName}:</strong>
                                <div>${formatMessageText(botText)}</div>
                                <div class="message-controls">
                                    <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${botIndex})">↻</button>
                                    <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${botIndex})">×</button>
                                    <div style="position: relative; display: inline-block;">
                                        <button class="branch-btn" onclick="toggleBranchDropdown(${botIndex}, event)">…</button>
                                        <div class="branch-menu" id="branch-menu-${botIndex}" style="display: none;"></div>
                                    </div>
                                </div>
                            </div>
                        `;
                       
                    }

                    if (isStreaming) {
                        await updateConversationList();
                    }
                } catch (error) {
                    if (error.name !== 'AbortError') {
                        messagesDiv.innerHTML = '';
                        history.history.forEach((msg, idx) => {
                            displayMessage(msg, msg.sender === 'user', idx);
                        });
                        alert('Failed to edit message: ' + error.message);
                    }
                } finally {
                    isStreaming = false;
                    document.getElementById('send-stop-btn').innerHTML = '&#10148;';
                    currentStreamController = null;
                    await updateBranchDropdown(index);
                }
            };
        }

        // Overwrite the existing deleteMessage definition in your <script>:

async function deleteMessage(index) {
    
        try {
            const response = await fetch('/delete_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message_index: index })
            });
            const data = await response.json();

            if (!data.success) {
                alert('Failed to delete message: ' + (data.error || 'Unknown error'));
                return;
            }

            // --- NEW LOGIC ---
            // Re-render the entire chat pane with the new history from the server.
            // This guarantees the client and server are perfectly in sync.
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML = '';
            data.history.forEach((msg, idx) => {
                displayMessage(msg, msg.sender === 'user', idx);
            });

        } catch (error) {
            console.error('Error deleting message:', error);
            alert('An error occurred while deleting the message.');
        }
    }
    


        
        async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    const pillContainer = document.getElementById('file-indicator-container');
    const activePills = pillContainer.querySelectorAll('.file-pill');

    if (!message && activePills.length === 0) return;

     // --- START OF MODIFIED LOGIC ---
    const attachments = [];
    const filenamesToMarkSent = [];
    for (const pill of activePills) {
        const filename = pill.dataset.filename;
        const filetype = pill.dataset.filetype;
        attachments.push({ filename, type: filetype });
        filenamesToMarkSent.push(filename);
    }
    
    // If there are files to send, mark them as sent on the backend first
    if (filenamesToMarkSent.length > 0) {
        try {
            await fetch('/mark_files_as_sent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: filenamesToMarkSent })
            });
        } catch (error) {
            alert("Could not confirm sent status of files. Please try again.");
            return; // Stop the message from sending if this fails
        }
    }
    // --- END OF MODIFIED LOGIC ---

    const messagesDiv = document.getElementById('chat-messages');
    const userIndex = messagesDiv.children.length;

    // --- IMPORTANT CHANGE ---
    // We now create a temporary message object to pass to displayMessage
    // This ensures the attachments are rendered immediately on the client side.
    const tempUserMsg = {
        text: message,
        profile_photo: currentUserPhoto,
        attachments: attachments
    };
    displayMessage(tempUserMsg, true, userIndex);
    // --- END ---
    
    input.value = '';
    input.style.height = '44px';

    pillContainer.innerHTML = '';
    updatePillContainerPosition();

    // --- START OF NEW LOGIC ---
    // Clear the tracking list and alert the user
    if (uploadedFileIndices.length > 0) {
        uploadedFileIndices = [];
        alert('list cleared');
    }
    // --- END OF NEW LOGIC ---

    const botIndex = userIndex + 1;
    const botMessageDiv = document.createElement('div');
    // ... (the creation of botMessageDiv and its innerHTML is unchanged) ...
    botMessageDiv.className = 'message bot-message';
    botMessageDiv.dataset.index = botIndex;
    // (innerHTML for bot message remains the same)
        if (!currentBotPhoto) {
        const convResponse = await fetch('/conversations');
        const convData = await convResponse.json();
        const convId = Object.keys(convData.conversations).find(id => convData.conversations[id].name === `Chat with ${currentBotName}`);
        currentBotPhoto = convId && convData.conversations[convId].bot_photo ? `/static/bot_photos/${convData.conversations[convId].bot_photo}` : 'https://via.placeholder.com/32';
    }

    botMessageDiv.innerHTML = `
        <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
        <div class="message-content">
            <strong>${currentBotName}:</strong>
            <div><div class="loading-spinner"></div> Generating Message...</div>
            <div class="message-controls">
                <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${botIndex})" style="display: none;">↻</button>
                <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${botIndex})" style="display: none;">×</button>
                <div style="position: relative; display: inline-block;">
                    <button class="branch-btn" onclick="toggleBranchDropdown(${botIndex}, event)" style="display: none;">…</button>
                    <div class="branch-menu" id="branch-menu-${botIndex}" style="display: none;"></div>
                </div>
            </div>
        </div>
    `;
    messagesDiv.appendChild(botMessageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;


    isStreaming = true;
    document.getElementById('send-stop-btn').innerHTML = '□';

    try {
        const controller = new AbortController();
        currentStreamController = controller;
        
        // --- IMPORTANT CHANGE: SEND ATTACHMENTS TO BACKEND ---
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: message, 
                attachments: attachments  // Send the attachments array
            }),
            signal: controller.signal
        });
        // --- END ---

        // The rest of the streaming logic remains exactly the same
        // ...
        if (!response.ok) {
            const errorData = await response.json();
            botMessageDiv.innerHTML = `
                <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
                <div class.message-content">
                    <strong>${currentBotName}:</strong>
                    <div>Error: ${errorData.error || 'Failed to process message'}</div>
                    <div class="message-controls">
                        <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${botIndex})">↻</button>
                        <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${botIndex})">×</button>
                        <div style="position: relative; display: inline-block;">
                            <button class="branch-btn" onclick="toggleBranchDropdown(${botIndex}, event)">…</button>
                            <div class="branch-menu" id="branch-menu-${botIndex}" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            `;
            throw new Error(errorData.error || 'Failed to process message');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let botText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done || !isStreaming) break;
            const chunk = decoder.decode(value, { stream: true });
            botText += chunk;
            botMessageDiv.innerHTML = `
                <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
                <div class="message-content">
                    <strong>${currentBotName}:</strong>
                    <div>${formatMessageText(botText)}</div>
                    <div class="message-controls">
                        <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${botIndex})">↻</button>
                        <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${botIndex})">×</button>
                        <div style="position: relative; display: inline-block;">
                            <button class="branch-btn" onclick="toggleBranchDropdown(${botIndex}, event)">…</button>
                            <div class="branch-menu" id="branch-menu-${botIndex}" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            `;
            
        }

        if (isStreaming) {
            const historyResponse = await fetch('/history');
            const history = await historyResponse.json();
            currentBotName = history.bot_name;
            const latestBotMsg = history.history.find(msg => msg.sender === 'bot');
            if (latestBotMsg && latestBotMsg.profile_photo) {
                currentBotPhoto = latestBotMsg.profile_photo;
            }
            document.getElementById('chat-header-title').textContent = currentBotName;
            await updateConversationList();
        }

    } catch (error) {
        if (error.name !== 'AbortError') {
             botMessageDiv.innerHTML = `
                <img src="${currentBotPhoto}" class="profile-pic" alt="${currentBotName}">
                <div class="message-content">
                    <strong>${currentBotName}:</strong>
                    <div>Error: ${error.message}</div>
                    <div class="message-controls">
                        <button class="regenerate-btn" title="Regenerate response" onclick="regenerateMessage(${botIndex})">↻</button>
                        <button class="delete-message-btn" title="Delete message" onclick="deleteMessage(${botIndex})">×</button>
                        <div style="position: relative; display: inline-block;">
                            <button class="branch-btn" onclick="toggleBranchDropdown(${botIndex}, event)">…</button>
                            <div class="branch-menu" id="branch-menu-${botIndex}" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            `;
        }
    } finally {
        isStreaming = false;
        document.getElementById('send-stop-btn').innerHTML = '&#10148;';
        currentStreamController = null;
        await updateBranchDropdown(botIndex);
    }
}

                async function updateConversationList() {
            const response = await fetch('/conversations');
            const data = await response.json(); // Now contains `conversations` and `bots`
            const listDiv = document.getElementById('conversation-list');
            listDiv.innerHTML = '';

            const botGroups = {}; // Key: Bot Name, Value: Array of conversation objects

            // Group all conversations
            Object.entries(data.conversations).forEach(([id, conv]) => {
                let botName = null;

                // Case 1: The conversation has a bot_id (the new, correct way)
                if (conv.bot_id && data.bots[conv.bot_id]) {
                    botName = data.bots[conv.bot_id].name;
                } 
                // Case 2: Fallback for older data without a bot_id
                else {
                    const match = conv.name.match(/Chat with (.+)/);
                    if (match) {
                        botName = match[1];
                    } else {
                        // For old, renamed conversations, group them by their own name.
                        // New conversations will not have this issue.
                        botName = conv.name; 
                    }
                }

                if (!botGroups[botName]) {
                    botGroups[botName] = [];
                }
                botGroups[botName].push({ id, name: conv.name });
            });

            // Render the grouped conversations
            Object.entries(botGroups).forEach(([botName, conversations]) => {
                const groupDiv = document.createElement('div');
                groupDiv.className = 'bot-group';
                
                const headerDiv = document.createElement('div');
                headerDiv.className = 'bot-header';
                headerDiv.innerHTML = `
                    ${botName}
                    <button class="new-chat-bot-btn" onclick="startNewChatWithBot('${botName}', event)">+</button>
                `;
                headerDiv.onclick = (e) => {
                    if (e.target.className !== 'new-chat-bot-btn') {
                        const convs = headerDiv.nextElementSibling;
                        convs.style.display = convs.style.display === 'block' ? 'none' : 'block';
                    }
                };

                const convsDiv = document.createElement('div');
                convsDiv.className = 'bot-conversations';
                
                conversations.forEach(conv => {
                    const convDiv = document.createElement('div');
                    convDiv.className = 'conversation-item';
                    convDiv.style.position = 'relative';
                    convDiv.onclick = () => loadConversation(conv.id);
                    convDiv.innerHTML = `
                        ${conv.name}
                        <button class="edit-convo-btn" onclick="renameChat('${conv.id}', event)">✎</button>
                        <button class="delete-convo-btn" onclick="deleteConversation('${conv.id}', event)">×</button>
                    `;
                    convsDiv.appendChild(convDiv);
                });

                groupDiv.appendChild(headerDiv);
                groupDiv.appendChild(convsDiv);
                listDiv.appendChild(groupDiv);
            });
        }

async function renameChat(convId, event) {
    event.stopPropagation();
    const newName = prompt("Enter new name for this chat:");
    if (!newName || !newName.trim()) return;

    try {
        const response = await fetch('/rename_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: convId, new_name: newName })
        });
        const data = await response.json();
        if (data.success) {
            await updateConversationList();
        } else {
            alert('Failed to rename chat: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Failed to rename chat');
    }
}

        // Replace your current deleteConversation with this:
async function deleteConversation(convId, event) {
    event.stopPropagation();
    if (!confirm('Are you sure you want to delete this chat?')) return;
    
    try {
        const response = await fetch('/delete_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: convId })
        });
        const data = await response.json();

        if (data.success) {
            // 1. Re-render the conversation list in the sidebar
            await updateConversationList();
            // 2. Clear the chat pane so you don’t see the deleted convo
            document.getElementById('chat-messages').innerHTML = '';
            // 3. Reset header title back to default
            document.getElementById('chat-header-title').textContent = 'Omni.ai';
        } else {
            alert('Failed to delete chat: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting chat:', error);
        alert('Failed to delete chat');
    }
}


        /* Updated functions to start new chat and load conversation without reloading the page or closing sidebar */

async function startNewChatWithBot(botName, event) {
    event.stopPropagation();
    try {
        const response = await fetch('/new_chat_with_bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bot_name: botName })
        });
        const data = await response.json();
        if (response.ok && data.success) {
            // 1. Re-render the conversation list
            await updateConversationList();
            // 2. Automatically load the newly created conversation
            await loadConversation(data.conversation_id);
            // 3. Keep sidebar open state intact
            const sidebar = document.getElementById('sidebar');
            if (!sidebar.classList.contains('open')) sidebar.classList.add('open');
        } else {
            alert('Failed to start new chat: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error starting new chat:', error);
        alert('Failed to start new chat');
    }
}

async function loadConversation(convId) {
    try {
        // 1. Tell server to switch context
        const resp = await fetch('/load_conversation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: convId })
        });
        const res = await resp.json();
        if (!res.success) throw new Error(res.error || 'Load failed');

        // 2. Re-render conversation list to reflect active selection
        await updateConversationList();

        // 3. Fetch history and set header
        const historyResponse = await fetch('/history');
        const historyData = await historyResponse.json();
        document.getElementById('chat-header-title').textContent = historyData.bot_name;

        // 4. Render messages
        const messagesDiv = document.getElementById('chat-messages');
        messagesDiv.innerHTML = '';
        historyData.history.forEach((msg, idx) => {
            displayMessage(msg, msg.sender === 'user', idx);
        });

        // 5. Ensure sidebar remains open on larger screens
        const sidebar = document.getElementById('sidebar');
        if (window.innerWidth >= 769 && !sidebar.classList.contains('open')) {
            sidebar.classList.add('open');
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
        alert('Failed to load conversation');
    }
}


        async function logout() {
            await fetch('/logout');
            window.location.href = '/';
        }

        async function switchBranch(branchId) {
            try {
                const response = await fetch('/switch_branch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ branch_id: branchId })
                });
                
                const data = await response.json();
                if (data.success) {
                    const messagesDiv = document.getElementById('chat-messages');
                    messagesDiv.innerHTML = '';
                    data.history.forEach((msg, idx) => {
                        displayMessage(msg, msg.sender === 'user', idx);
                    });
                }
            } catch (error) {
                alert('Failed to switch branch');
            }
        }

        window.onload = async () => {
    await updateConversationList();

    const photoResponse = await fetch("/get-user-photo");
    const photoData = await photoResponse.json();
    currentUserPhoto = photoData.photo || 'https://via.placeholder.com/32';

    const historyResponse = await fetch('/history');
    const history = await historyResponse.json();
    currentBotName = history.bot_name;
    document.getElementById('chat-header-title').textContent = currentBotName;

    const botMsg = history.history.find(msg => msg.sender === 'bot');
    if (botMsg && botMsg.profile_photo) {
        currentBotPhoto = botMsg.profile_photo;
    } else {
        const convResponse = await fetch('/conversations');
        const convData = await convResponse.json();
        const convId = Object.keys(convData.conversations)[0];
        currentBotPhoto = convId && convData.conversations[convId].bot_photo ? `/static/bot_photos/${convData.conversations[convId].bot_photo}` : 'https://via.placeholder.com/32';
    }

    history.history.forEach((msg, idx) => {
        displayMessage(msg, msg.sender === 'user', idx);
    });

    // --- PASTE THE FOLLOWING BLOCK HERE ---

    // Recreate pills for any files that are already active on page load
    try {
        // Fetch and render active PDF pills
        const pdfsResponse = await fetch('/get_pdfs');
        const pdfsData = await pdfsResponse.json();
        pdfsData.pdfs.forEach(pdf => {
            if (pdf.active) {
                createFileIndicatorPill(pdf.name, 'pdf', null);
            }
        });

        // Fetch and render active Image pills
        const imagesResponse = await fetch('/get_images');
        const imagesData = await imagesResponse.json();
        imagesData.images.forEach(image => {
            if (image.active) {
                createFileIndicatorPill(image.name, 'image', image.number);
            }
        });
    } catch (error) {
        console.error("Error recreating file pills on load:", error);
    }
    // --- END OF NEW BLOCK ---
    updatePillContainerPosition(); // Initial position adjustment
};
// FIX: Replace the last event listener in your script with this one
document.addEventListener('click', (event) => {
    // If the click is not on a branch button AND not inside a branch menu, close all menus.
    // stopPropagation() in the toggle function prevents this from firing when a button is clicked.
    if (!event.target.closest('.branch-btn') && !event.target.closest('.branch-menu')) {
        document.querySelectorAll('.branch-menu').forEach(menu => {
            menu.style.display = 'none';
        });
        document.querySelectorAll('.message.menu-active').forEach(msg => {
            msg.classList.remove('menu-active');
        });
    }
});
    </script>
</body>
</html>
"""

bots_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Browse Bots</title>
    <style>
        :root {
            --primary-bg: #121212;
            --card-bg: #242424;
            --card-hover-bg: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent: #4d84ff;
            --border-radius: 12px;
        }
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--primary-bg);
            color: var(--text-primary);
        }
        
        .page-header {
            display: flex;
            align-items: center;
            padding: 20px 28px;
            max-width: 1400px;
            margin: 0 auto;
            gap: 20px;
        }
        
        .back-button {
            background: var(--card-bg);
            color: var(--text-primary);
            border: none;
            border-radius: var(--border-radius);
            padding: 10px 15px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .back-button:hover {
            background: var(--card-hover-bg);
            transform: translateX(-2px);
        }
        
        .search-container {
            flex: 1;
            max-width: 500px;
        }
        
        .search-bar {
            width: 100%;
            padding: 12px 16px;
            border-radius: var(--border-radius);
            border: 1px solid rgba(255, 255, 255, 0.1);
            background-color: var(--card-bg);
            color: var(--text-primary);
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
        }
        
        .search-bar:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(77, 132, 255, 0.2);
        }
        
        .bot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            padding: 0 28px 28px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .bot-card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 18px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .bot-card:hover {
            transform: translateY(-5px);
            background: var(--card-hover-bg);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            border-color: var(--accent);
        }
        
        .bot-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: var(--border-radius);
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }
        
        .bot-name {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-primary);
        }
        
        .bot-description {
            color: var(--text-secondary);
            font-size: 0.95em;
            line-height: 1.5;
        }
        
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <header class="page-header">
        <button class="back-button" onclick="goToMainPage()">
            ← Back
        </button>
        <div class="search-container">
            <input type="text" class="search-bar" placeholder="Search bots by name..." id="searchInput" oninput="searchBots()">
        </div>
    </header>
    
    <div class="bot-grid" id="botGrid">
        {% for bot in bots %}
    <div class="bot-card" onclick="startChatWithBot('{{ bot.id }}')" data-name="{{ bot.name | lower }}">
    {% if bot.photo %}
    <img src="/static/bot_photos/{{ bot.photo }}" class="bot-image" alt="{{ bot.name }}">
    {% endif %}
    <div class="bot-name">{{ bot.name }}</div>
    <div class="bot-description">{{ bot.description }}</div>
    {% if bot.creator == session['user_id'] %}
    <div class="delete-bot-btn" onclick="event.stopPropagation(); deleteBot('{{ bot.id }}', '{{ bot.name }}')">Delete Bot</div>
    {% endif %}
</div>
        {% endfor %}
    </div>
    <script>
        async function startChatWithBot(botId) {
            const response = await fetch(`/start-bot-chat/${botId}`);
            if(response.ok) {
                window.location.href = '/';
            } else {
                alert('Error starting chat');
            }
        }
        
        function goToMainPage() {
            window.location.href = '/';
        }
        
        function searchBots() {
            const searchInput = document.getElementById('searchInput');
            const searchTerm = searchInput.value.toLowerCase();
            const botGrid = document.getElementById('botGrid');
            const botCards = Array.from(document.querySelectorAll('.bot-card'));
            
            // First, make all bots visible again
            botCards.forEach(card => {
                card.classList.remove('hidden');
            });
            
            if (searchTerm.trim() === '') {
                // If search is empty, restore original order
                botGrid.innerHTML = '';
                botCards.forEach(card => {
                    botGrid.appendChild(card);
                });
                return;
            }
            
            // Sort the cards based on how well they match the search term
            botCards.sort((a, b) => {
                const nameA = a.getAttribute('data-name');
                const nameB = b.getAttribute('data-name');
                
                // Exact matches at the top
                const exactMatchA = nameA === searchTerm;
                const exactMatchB = nameB === searchTerm;
                
                if (exactMatchA && !exactMatchB) return -1;
                if (!exactMatchA && exactMatchB) return 1;
                
                // Then starts-with matches
                const startsWithA = nameA.startsWith(searchTerm);
                const startsWithB = nameB.startsWith(searchTerm);
                
                if (startsWithA && !startsWithB) return -1;
                if (!startsWithA && startsWithB) return 1;
                
                // Then contains matches
                const containsA = nameA.includes(searchTerm);
                const containsB = nameB.includes(searchTerm);
                
                if (containsA && !containsB) return -1;
                if (!containsA && containsB) return 1;
                
                // Hide non-matches
                if (!containsA) a.classList.add('hidden');
                if (!containsB) b.classList.add('hidden');
                
                return 0;
            });
            
            // Re-add the sorted cards to the grid
            botGrid.innerHTML = '';
            botCards.forEach(card => {
                botGrid.appendChild(card);
            });
        }
        //new function to confirm deletion of bot
    async function confirmDeleteBot(botId, botName) {
        if (confirm(`Are you sure you want to delete bot "${botName}"? This will also delete all conversations with this bot.`)) {
        try {
            const response = await fetch(`/delete-bot/${botId}`, {
                method: 'POST',
            });
            
            const data = await response.json();
            if (response.ok) {
                alert('Bot deleted successfully');
                // Remove the bot card from the UI
                document.querySelector(`.bot-card[onclick*="${botId}"]`).remove();
            } else {
                alert(data.error || 'Failed to delete bot');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Bot deleted. Refresh screen to see changes.');
        }
    }
}
        async function deleteBot(botId, botName) {
    if (confirm(`Are you sure you want to delete bot "${botName}"? This will also delete all conversations with this bot.`)) {
        try {
            const response = await fetch(`/delete-bot/${botId}`, {
                method: 'POST',
            });
            
            const data = await response.json();
            if (response.ok) {
                alert('Bot deleted successfully');
                // Remove the bot card from the UI
                document.querySelector(`.bot-card[onclick*="${botId}"]`).remove();
            } else {
                alert(data.error || 'Failed to delete bot');
            }
        } catch (error) { //Used to have an error. Fixed it! Can properly delete bots now. 
            console.error('Error:', error);
            alert('Bot deleted. Refresh screen to see changes');
        }
    }
}
    </script>
</body>
</html>
"""

login_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial; background: #1a1a1a; color: white; }
        .auth-container { max-width: 300px; margin: 100px auto; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tabs button { flex: 1; padding: 10px; background: #333; border: none; color: white; cursor: pointer; }
        .tabs button:hover { background: #444; }
        .form { display: flex; flex-direction: column; gap: 10px; }
        input { padding: 10px; border: 1px solid #444; background: #222; color: white; border-radius: 4px; }
        button { padding: 10px; background: #4CAF50; border: none; color: white; cursor: pointer; }
    </style>
</head>
<body>
    <div class="auth-container">
        <div class="tabs">
            <button onclick="showTab('login')">Login</button>
            <button onclick="showTab('register')">Register</button>
        </div>
        <div id="login-form" class="form">
            <input type="text" id="login-username" placeholder="Username">
            <input type="password" id="login-password" placeholder="Password">
            <button onclick="login()">Login</button>
        </div>
        <div id="register-form" class="form" style="display:none;">
            <input type="text" id="register-username" placeholder="Username">
            <input type="password" id="register-password" placeholder="Password">
            <button onclick="register()">Register</button>
        </div>
    </div>
    <script>
        function showTab(tab) {
            document.getElementById('login-form').style.display = tab === 'login' ? 'flex' : 'none';
            document.getElementById('register-form').style.display = tab === 'register' ? 'flex' : 'none';
        }
        async function login() {
            const username = document.getElementById('login-username').value;
            const password = document.getElementById('login-password').value;
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json();
            if (data.success) window.location.reload();
            else alert(data.error || 'Login failed');
        }
        async function register() {
            const username = document.getElementById('register-username').value;
            const password = document.getElementById('register-password').value;
            const response = await fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json();
            if (data.success) {
                alert('Registration successful! Please login.');
                showTab('login');
            } else alert(data.error || 'Registration failed');
        }
    </script>
</body>
</html>
"""
create_bot_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Create Bot</title>
    <style>
        body { background: #181818; color: white; padding: 20px; font-family: Arial, sans-serif; }
        .back-button {
            background: #333;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .back-button:hover {
            background: #444;
        }
        .form-container { max-width: 500px; margin: 0 auto; }
        .form-group { margin-bottom: 15px; }
        input, textarea { width: 100%; padding: 10px; background: #242424; border: 1px solid #3d3d3d; color: white; border-radius: 4px; }
        button[type="submit"] { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; margin-top: 10px; border-radius: 4px; }
        button[type="submit"]:hover { background: #45a049; }
        /* Cropping Modal Styles */
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.8); z-index: 1000; justify-content: center; align-items: center; }
        .modal-content { background: #242424; padding: 20px; border-radius: 8px; max-width: 600px; width: 90%; }
        #crop-image { max-width: 100%; max-height: 400px; display: block; margin: 0 auto; }
        .modal-buttons { margin-top: 10px; display: flex; justify-content: space-between; }
    </style>
    <!-- Include Cropper.js CSS and JS -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js"></script>
</head>
<body>
    <!-- Back to Chat Button -->
    <button class="back-button" onclick="window.location.href='/'">← Back to Chat</button>

    <div class="form-container">
        <h2>Create New Bot</h2>
        <form id="botForm" enctype="multipart/form-data">
            <div class="form-group">
                <label>Bot Name:</label>
                <input type="text" name="name" required>
            </div>
            <div class="form-group">
                <label>Description:</label>
                <textarea name="description" required rows="3"></textarea>
            </div>
            <div class="form-group">
                <label>System Prompt:</label>
                <textarea name="system_prompt" required rows="5"></textarea>
            </div>
            <div class="form-group">
                <label>Bot Photo:</label>
                <input type="file" id="photo-input" name="photo" accept="image/*">
            </div>
            <button type="submit">Create Bot</button>
        </form>
    </div>

    <!-- Cropping Modal -->
    <div id="crop-modal" class="modal">
        <div class="modal-content">
            <h3>Crop Your Image</h3>
            <img id="crop-image" src="">
            <div class="modal-buttons">
                <button id="cancel-crop" style="background: #ff4444; border: none; color: white; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button id="crop-submit" style="background: #4CAF50; border: none; color: white; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Crop & Submit</button>
            </div>
        </div>
    </div>

    <script>
        let cropper;
        const form = document.getElementById('botForm');
        const photoInput = document.getElementById('photo-input');
        const cropModal = document.getElementById('crop-modal');
        const cropImage = document.getElementById('crop-image');
        const cancelCrop = document.getElementById('cancel-crop');
        const cropSubmit = document.getElementById('crop-submit');
        let croppedBlob = null;

        photoInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    cropImage.src = event.target.result;
                    cropModal.style.display = 'flex';
                    if (cropper) cropper.destroy();
                    cropper = new Cropper(cropImage, {
                        aspectRatio: 1,
                        viewMode: 1,
                        autoCropArea: 0.8,
                        movable: true,
                        zoomable: true,
                        scalable: false,
                        background: false,
                    });
                };
                reader.readAsDataURL(file);
            }
        });

        cancelCrop.addEventListener('click', () => {
            cropModal.style.display = 'none';
            photoInput.value = '';
            if (cropper) cropper.destroy();
        });

        cropSubmit.addEventListener('click', () => {
            if (cropper) {
                cropper.getCroppedCanvas({ width: 200, height: 200 }).toBlob((blob) => {
                    croppedBlob = blob;
                    cropModal.style.display = 'none';
                    cropper.destroy();
                    submitForm();
                }, 'image/jpeg', 0.9);
            }
        });

        async function submitForm() {
            const formData = new FormData(form);
            if (croppedBlob) {
                formData.delete('photo');
                formData.append('photo', croppedBlob, 'cropped_image.jpg');
            }
            const response = await fetch('/create-bot', { method: 'POST', body: formData });
            if (response.ok) {
                alert('Bot created successfully!');
                window.location.href = '/bots';
            } else {
                alert('Error creating bot');
            }
        }

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            if (photoInput.files.length > 0 && !croppedBlob) return;
            submitForm();
        });
    </script>
</body>
</html>
"""

my_bots_html = """
<!DOCTYPE html>
<html>
<head>
    <title>My Bots</title>
    <style>
        :root {
            --primary-bg: #121212;
            --card-bg: #242424;
            --card-hover-bg: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent: #4d84ff;
            --border-radius: 12px;
            --danger: #ff4444;
        }
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--primary-bg);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .page-header {
            padding: 20px 28px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .back-button {
            background: var(--card-bg);
            color: var(--text-primary);
            border: none;
            border-radius: var(--border-radius);
            padding: 10px 15px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .back-button:hover {
            background: var(--card-hover-bg);
            transform: translateX(-2px);
        }
        
        .bot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 0 28px 28px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .bot-card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 18px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
        }
        
        .bot-card:hover {
            transform: translateY(-5px);
            background: var(--card-hover-bg);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        
        .bot-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: var(--border-radius);
            margin-bottom: 12px;
        }
        
        .bot-name {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .bot-description {
            color: var(--text-secondary);
            font-size: 0.95em;
            line-height: 1.5;
            margin-bottom: 10px;
        }
        
        .edit-form {
            display: none;
            flex-direction: column;
            gap: 10px;
        }
        
        .edit-form input, .edit-form textarea {
            width: 100%;
            padding: 8px;
            background: #2d2d2d;
            border: 1px solid #3d3d3d;
            color: white;
            border-radius: 4px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .edit-btn, .delete-btn, .save-btn, .cancel-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .edit-btn {
            background: var(--accent);
            color: white;
        }
        
        .delete-btn {
            background: var(--danger);
            color: white;
        }
        
        .save-btn {
            background: #4CAF50;
            color: white;
        }
        
        .cancel-btn {
            background: #666;
            color: white;
        }
        
        .no-bots {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        .no-bots a {
            color: var(--accent);
            text-decoration: none;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js"></script>
</head>
<body>
    <div class="page-header">
        <button class="back-button" onclick="window.location.href='/'">← Back to Chat</button>
    </div>
    
    <div class="bot-grid" id="bot-grid">
        {% if my_bots %}
            {% for bot in my_bots %}
                <div class="bot-card" data-bot-id="{{ bot.id }}">
                    <img src="{{ '/static/bot_photos/' + bot.photo if bot.photo else 'https://via.placeholder.com/200' }}" class="bot-image" alt="{{ bot.name }}">
                    <div class="bot-info">
                        <div class="bot-name">{{ bot.name }}</div>
                        <div class="bot-description">{{ bot.description }}</div>
                        <div class="button-group">
                            <button class="edit-btn" onclick="toggleEditForm('{{ bot.id }}')">Edit</button>
                            <button class="delete-btn" onclick="deleteBot('{{ bot.id }}', '{{ bot.name }}')">Delete</button>
                        </div>
                    </div>
                    <form class="edit-form" id="edit-form-{{ bot.id }}" enctype="multipart/form-data">
                        <input type="text" name="name" value="{{ bot.name }}" required>
                        <textarea name="description" rows="3" required>{{ bot.description }}</textarea>
                        <textarea name="system_prompt" rows="5" required>{{ bot.system_prompt }}</textarea>
                        <input type="file" name="photo" accept="image/*" onchange="handlePhotoChange('{{ bot.id }}', event)">
                        <div class="button-group">
                            <button type="button" class="save-btn" onclick="saveBotChanges('{{ bot.id }}')">Save</button>
                            <button type="button" class="cancel-btn" onclick="toggleEditForm('{{ bot.id }}')">Cancel</button>
                        </div>
                    </form>
                </div>
            {% endfor %}
        {% else %}
            <div class="no-bots">
                <h2>You haven't created any bots yet!</h2>
                <p><a href="/create-bot">Create your first bot now</a></p>
            </div>
        {% endif %}
    </div>

    <div id="crop-modal" class="modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center;">
        <div class="modal-content" style="background: #242424; padding: 20px; border-radius: 8px; max-width: 600px; width: 90%;">
            <h3>Crop Your Image</h3>
            <img id="crop-image" src="" style="max-width: 100%; max-height: 400px; display: block; margin: 0 auto;">
            <div class="modal-buttons" style="margin-top: 10px; display: flex; justify-content: space-between;">
                <button id="cancel-crop" style="background: #ff4444; padding: 8px 16px; border: none; color: white; border-radius: 4px;">Cancel</button>
                <button id="crop-submit" style="background: #4CAF50; padding: 8px 16px; border: none; color: white; border-radius: 4px;">Crop & Submit</button>
            </div>
        </div>
    </div>

    <script>
        let cropper;
        let currentBotId;
        let croppedBlob = null;

        function toggleEditForm(botId) {
            const card = document.querySelector(`.bot-card[data-bot-id="${botId}"]`);
            const info = card.querySelector('.bot-info');
            const form = card.querySelector('.edit-form');
            const isEditing = form.style.display === 'flex';
            info.style.display = isEditing ? 'block' : 'none';
            form.style.display = isEditing ? 'none' : 'flex';
        }

        function handlePhotoChange(botId, event) {
            const file = event.target.files[0];
            if (file) {
                currentBotId = botId;
                const reader = new FileReader();
                reader.onload = (e) => {
                    const cropImage = document.getElementById('crop-image');
                    cropImage.src = e.target.result;
                    document.getElementById('crop-modal').style.display = 'flex';
                    if (cropper) cropper.destroy();
                    cropper = new Cropper(cropImage, {
                        aspectRatio: 1,
                        viewMode: 1,
                        autoCropArea: 0.8,
                        movable: true,
                        zoomable: true,
                        scalable: false,
                        background: false,
                    });
                };
                reader.readAsDataURL(file);
            }
        }

        document.getElementById('cancel-crop').addEventListener('click', () => {
            document.getElementById('crop-modal').style.display = 'none';
            const form = document.getElementById(`edit-form-${currentBotId}`);
            form.querySelector('input[name="photo"]').value = '';
            if (cropper) cropper.destroy();
            croppedBlob = null;
        });

        document.getElementById('crop-submit').addEventListener('click', () => {
            if (cropper) {
                cropper.getCroppedCanvas({
                    width: 200,
                    height: 200,
                }).toBlob((blob) => {
                    croppedBlob = blob;
                    document.getElementById('crop-modal').style.display = 'none';
                    cropper.destroy();
                }, 'image/jpeg', 0.9);
            }
        });

        async function saveBotChanges(botId) {
            const form = document.getElementById(`edit-form-${botId}`);
            const formData = new FormData(form);
            if (croppedBlob) {
                formData.delete('photo');
                formData.append('photo', croppedBlob, 'cropped_image.jpg');
            }

            try {
                const response = await fetch(`/edit-bot/${botId}`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    window.location.reload();
                } else {
                    alert(data.error || 'Failed to update bot');
                }
            } catch (error) {
                alert('Error updating bot: ' + error.message);
            }
        }

        async function deleteBot(botId, botName) {
            if (!confirm(`Are you sure you want to delete "${botName}"? This will also delete all associated conversations.`)) return;
            try {
                const response = await fetch(`/delete-bot/${botId}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (data.success) {
                    document.querySelector(`.bot-card[data-bot-id="${botId}"]`).remove();
                    if (!document.querySelector('.bot-card')) {
                        document.getElementById('bot-grid').innerHTML = `
                            <div class="no-bots">
                                <h2>You haven't created any bots yet!</h2>
                                <p><a href="/create-bot">Create your first bot now</a></p>
                            </div>
                        `;
                    }
                } else {
                    alert(data.error || 'Failed to delete bot');
                }
            } catch (error) {
                alert('Error deleting bot: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

user_profile_html = """
<!DOCTYPE html>
<html>
<head>
    <title>User Profile</title>
    <style>
        :root {
            --primary-bg: #1a1a1a;
            --card-bg: #252525;
            --card-hover-bg: #303030;
            --text-primary: #e0e0e0;
            --text-secondary: #909090;
            --border-color: #404040;
            --button-bg: #353535;
            --button-hover: #454545;
            --danger: #803030;
            --danger-hover: #9b3b3b;
            --border-radius: 8px;
        }
        
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--primary-bg);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .container {
            max-width: 500px;
            margin: 0 auto;
        }
        
        .back-button {
            background: var(--button-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 8px 12px;
            cursor: pointer;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .back-button:hover {
            background: var(--button-hover);
        }
        
        .profile-card {
            background: var(--card-bg);
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            text-align: center;
        }
        
        .profile-image {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            object-fit: cover;
            margin: 0 auto 15px;
            display: block;
            border: 1px solid var(--border-color);
        }
        
        .username {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: var(--text-primary);
        }
        
        .form-section {
            margin: 20px 0;
        }
        
        .form-section label {
            display: block;
            text-align: left;
            margin-bottom: 5px;
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .form-section input[type="text"] {
            width: 100%;
            padding: 8px;
            background: var(--primary-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 14px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        button {
            background: var(--button-bg);
            color: var(--text-primary);
            padding: 8px 16px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            width: 100%;
        }
        
        button:hover {
            background: var(--button-hover);
        }
        
        .delete-account-btn {
            background: var(--danger);
            border-color: var(--danger);
        }
        
        .delete-account-btn:hover {
            background: var(--danger-hover);
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background: var(--card-bg);
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            max-width: 600px;
            width: 90%;
        }
        
        #crop-image {
            max-width: 100%;
            max-height: 400px;
            display: block;
            margin: 0 auto;
        }
        
        .modal-buttons {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            gap: 10px;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js"></script>
</head>
<body>
    <div class="container">
        <button class="back-button" onclick="window.location.href='/'">← Back to Chat</button>
        <div class="profile-card">
            <img src="{{ profile_photo if profile_photo else 'https://via.placeholder.com/120' }}" class="profile-image" alt="Profile">
            <div class="username">{{ username }}</div>
            
            <div class="form-section">
                <label>Change Profile Picture</label>
                <button onclick="document.getElementById('photo-input').click()">Upload New Picture</button>
                <input type="file" id="photo-input" accept="image/*" style="display: none;">
            </div>
            
            <div class="form-section">
                <label>Change Username</label>
                <input type="text" id="new-username" placeholder="New username">
                <div class="button-group">
                    <button onclick="changeUsername()">Save</button>
                    <button onclick="document.getElementById('new-username').value = ''">Clear</button>
                </div>
            </div>
            
            <div class="form-section">
                <button class="delete-account-btn" onclick="deleteAccount()">Delete Account</button>
            </div>
        </div>
    </div>

    <div id="crop-modal" class="modal">
        <div class="modal-content">
            <h3 style="margin-top: 0; color: var(--text-primary);">Crop Profile Picture</h3>
            <img id="crop-image" src="">
            <div class="modal-buttons">
                <button id="cancel-crop" style="background: var(--danger); border-color: var(--danger);">Cancel</button>
                <button id="crop-submit" style="background: var(--button-bg); border-color: var(--border-color);">Save</button>
            </div>
        </div>
    </div>

    <script>
        let cropper;
        const photoInput = document.getElementById('photo-input');

        async function changeUsername() {
            const newUsername = document.getElementById('new-username').value.trim();
            if (!newUsername) {
                alert('Please enter a new username');
                return;
            }

            try {
                const response = await fetch('/change-username', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ new_username: newUsername })
                });
                const data = await response.json();
                if (data.success) {
                    window.location.reload();
                } else {
                    alert(data.error || 'Failed to change username');
                }
            } catch (error) {
                alert('Error changing username');
            }
        }

        async function deleteAccount() {
            if (!confirm('Are you sure you want to delete your account? This action cannot be undone and will remove all your data.')) return;
            
            try {
                const response = await fetch('/delete-account', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await response.json();
                if (data.success) {
                    window.location.href = '/';
                } else {
                    alert(data.error || 'Failed to delete account');
                }
            } catch (error) {
                alert('Error deleting account');
            }
        }

        photoInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    document.getElementById('crop-image').src = event.target.result;
                    document.getElementById('crop-modal').style.display = 'flex';
                    if (cropper) cropper.destroy();
                    cropper = new Cropper(document.getElementById('crop-image'), {
                        aspectRatio: 1,
                        viewMode: 1,
                        autoCropArea: 0.8,
                        movable: true,
                        zoomable: true,
                        scalable: false,
                        background: false,
                        cropBoxResizable: true,
                        dragMode: 'move',
                        crop(event) {
                            const cropBoxData = cropper.getCropBoxData();
                            cropper.setCropBoxData({
                                width: Math.min(cropBoxData.width, cropBoxData.height),
                                height: Math.min(cropBoxData.width, cropBoxData.height)
                            });
                        }
                    });
                };
                reader.readAsDataURL(file);
            }
        });

        document.getElementById('cancel-crop').addEventListener('click', () => {
            document.getElementById('crop-modal').style.display = 'none';
            photoInput.value = '';
            if (cropper) cropper.destroy();
        });

        document.getElementById('crop-submit').addEventListener('click', async () => {
            if (cropper) {
                const canvas = cropper.getCroppedCanvas({
                    width: 120,
                    height: 120,
                });
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('photo', blob, 'profile.jpg');

                    try {
                        const response = await fetch('/upload-profile-photo', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        if (data.success) {
                            window.location.reload();
                        } else {
                            alert('Failed to upload profile picture: ' + (data.error || 'Unknown error'));
                        }
                    } catch (error) {
                        alert('Error uploading profile picture');
                    }

                    document.getElementById('crop-modal').style.display = 'none';
                    cropper.destroy();
                }, 'image/jpeg', 0.9);
            }
        });
    </script>
</body>
</html>
"""

# Add this temporary migration code before starting the Flask app
if __name__ == '__main__':
    # Migrate existing branches to include action_message_index
    for user_id, convs in all_conversations.items():
        for conv_id, conv in convs.items():
            for branch_id, branch in conv.get('branches', {}).items():
                if 'action_message_index' not in branch:
                    # Set action_message_index to parent_message_index for existing branches
                    branch['action_message_index'] = branch.get('parent_message_index')
    
    # Save migrated data
    with open(SAVE_FILE, "w") as f:
        json.dump(all_conversations, f)
    app.run()
