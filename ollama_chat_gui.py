import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import requests
import json
import time
from datetime import timedelta
import os
import numpy as np
from PyPDF2 import PdfReader

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

#######################################################################################
MODEL_NAME = "deepseek-r1:8b"
TEMPERATURE = 0.6       # lower temperature for more precise response vs higher for more creative response
CTX_WINDOW = 16384	    # CONTEXT WINDOW -> min 2048 - max 131072 (more context windows requires more VRAM!!!)
KEEP_ALIVE = 0
#######################################################################################

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
document_chunks = []
document_embeddings = None
conversation_messages = []
context_system_message = None
loaded_files = []

CONTEXT_FILE = "chat_context.txt"
if os.path.exists(CONTEXT_FILE):
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        saved_context = f.read().strip()
        if saved_context:
            context_system_message = SystemMessage(content=saved_context)

def load_document_file(filepath):
    global document_chunks, document_embeddings, embedder
    ext = os.path.splitext(filepath)[1].lower()
    text_content = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text_content = f.read()
        else:
            messagebox.showerror("Error", f"File format {ext} not supported!")
            return
    except Exception as e:
        messagebox.showerror("Error", f"File read error: {e}")
        return

    chunk_size = 1000
    overlap = 100
    chunks = []
    for i in range(0, len(text_content), chunk_size - overlap):
        chunk = text_content[i:i+chunk_size]
        last_newline = chunk.rfind("\n")
        if last_newline != -1 and last_newline > len(chunk) * 0.8:
            chunk = chunk[:last_newline]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(text_content):
            break

    if not chunks:
        messagebox.showwarning("WARNING!", "No useful text were found...")
        return

    try:
        new_embeddings = embedder.embed_documents(chunks)
    except Exception as e:
        messagebox.showerror("Error", f"Embedding evaluation errors: {e}")
        return
    new_embeddings = np.array(new_embeddings)

    if document_embeddings is None:
        document_embeddings = new_embeddings
    else:
        document_embeddings = np.vstack([document_embeddings, new_embeddings])
    document_chunks.extend(chunks)

    filename = os.path.basename(filepath)
    if filepath not in loaded_files:
        loaded_files.append(filepath)
        add_file_to_list(filename, filepath)

    messagebox.showinfo("Document loaded", f"'{filename}' with {len(chunks)} segmets.")

def retrieve_relevant_chunks(query, top_k=3):
    global document_chunks, document_embeddings, embedder
    if document_embeddings is None or len(document_chunks) == 0:
        return []
    query_embedding = embedder.embed_query(query)
    doc_emb = document_embeddings
    norms = np.linalg.norm(doc_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed_doc_emb = doc_emb / norms
    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        return []
    normed_query = query_embedding / q_norm
    similarities = np.dot(normed_doc_emb, normed_query)
    top_idx = similarities.argsort()[-top_k:][::-1]
    top_chunks = [document_chunks[i] for i in top_idx]
    return top_chunks

root = tk.Tk()
root.title("OLLAMA CHAT")
root.state('zoomed')

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(main_frame, width=200, bg="#2b2b2b")
right_frame.pack(side=tk.RIGHT, fill=tk.Y)

file_label = tk.Label(right_frame, text="Files loaded", fg="white", bg="#2b2b2b")
file_label.pack(pady=(10, 0))

file_list_frame = tk.Frame(right_frame, bg="#2b2b2b")
file_list_frame.pack(fill=tk.Y, padx=5, pady=5)

def add_file_to_list(filename, file_path):
    row_frame = tk.Frame(file_list_frame, bg="#2b2b2b")
    row_frame.pack(fill=tk.X, pady=2)
    lbl = tk.Label(row_frame, text=filename, anchor="w", bg="#2b2b2b", fg="white")
    lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    btn = tk.Button(row_frame, text="x", command=lambda: remove_file(file_path, row_frame))
    btn.pack(side=tk.RIGHT)

def remove_file(file_path, frame):
    if file_path in loaded_files:
        loaded_files.remove(file_path)
    frame.destroy()

chat_label = tk.Label(left_frame, text="LLM ANSWER", fg="white", bg="#1e1e1e")
chat_label.pack(anchor="w", padx=5)
chat_display = ScrolledText(left_frame, wrap=tk.WORD, state='disabled', width=80, height=20)
chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
chat_display.tag_config("user", foreground="blue")
chat_display.tag_config("assistant", foreground="green")

def append_to_chat(role, message):
    chat_display.configure(state='normal')
    tag = "user" if role == "user" else "assistant"
    chat_display.insert(tk.END, f"{role.capitalize()}: {message}\n", (tag,))
    chat_display.configure(state='disabled')
    chat_display.yview(tk.END)

context_label = tk.Label(left_frame, text="LLM CONTEXT", fg="white", bg="#1e1e1e")
context_label.pack(anchor="w", padx=5)
context_input = ScrolledText(left_frame, wrap=tk.WORD, height=5, width=80)
if context_system_message:
    context_input.insert("1.0", context_system_message.content)
context_input.pack(fill=tk.BOTH, expand=False, padx=5, pady=(0, 5))

def apply_context():
    global context_system_message
    content = context_input.get("1.0", tk.END).strip()
    if content:
        context_system_message = SystemMessage(content=content)
        with open("chat_context.txt", "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Context applyed", "Context applyed and saved successfully.")

context_btn = tk.Button(left_frame, text="APPLY CONTEXT", command=apply_context)
context_btn.pack(padx=5, pady=(0,5))

input_frame = tk.Frame(left_frame)
input_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

user_input_label = tk.Label(input_frame, text="USER REQUEST", fg="white", bg="#1e1e1e")
user_input_label.pack(anchor="nw", padx=5)
user_input = ScrolledText(input_frame, wrap=tk.WORD, height=5, width=80)
user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

def send_message():
    global conversation_messages
    user_query = user_input.get("1.0", tk.END).strip()
    if user_query == "":
        return
    start_time = time.time()
    append_to_chat("user", user_query)
    conversation_messages.append(HumanMessage(content=user_query))
    user_input.delete("1.0", tk.END)
    messages_to_send = []
    if context_system_message:
        messages_to_send.append(context_system_message)
    relevant_chunks = retrieve_relevant_chunks(user_query) if document_embeddings is not None else []
    if relevant_chunks:
        context_text = "\n---\n".join(relevant_chunks)
        context_msg = SystemMessage(content=f"Infos from loaded documents:\n{context_text}")
        messages_to_send.append(context_msg)
    messages_to_send.extend(conversation_messages)
    try:
        chat = ChatOllama(model=MODEL_NAME,
                            temperature=TEMPERATURE,
                            num_ctx=CTX_WINDOW,
                            keep_alive=KEEP_ALIVE)
        reply = chat.invoke(messages_to_send).content.strip()
    except Exception as e:
        messagebox.showerror("Error", f"Model call generic error: {e}")
        return
    append_to_chat("assistant", reply)
    conversation_messages.append(AIMessage(content=reply))
    elapsed_time = time.time() - start_time
    append_to_chat("assistant", f"(answer time: {str(timedelta(seconds=int(elapsed_time)))})")

def load_conversation():
    global conversation_messages
    load_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if not load_path:
        return
    try:
        with open(load_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        conversation_messages = []
        chat_display.configure(state='normal')
        chat_display.delete("1.0", tk.END)
        for entry in loaded:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "system":
                continue
            if role == "user":
                conversation_messages.append(HumanMessage(content=content))
                chat_display.insert(tk.END, f"User: {content}\n", ("user",))
            elif role == "ai":
                conversation_messages.append(AIMessage(content=content))
                chat_display.insert(tk.END, f"Assistant: {content}\n", ("assistant",))
        chat_display.configure(state='disabled')
        chat_display.yview(tk.END)
        messagebox.showinfo("Last conversasion loaded", f"Conversasion loaded from: {load_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Last conversasion loading error: {e}")

def save_conversation():
    if not conversation_messages:
        messagebox.showinfo("SAVE", "No conversation to save!")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([{"role": msg.type, "content": msg.content} for msg in conversation_messages], f, indent=2, ensure_ascii=False)
        messagebox.showinfo("CONVERSATION SAVED", f"Conversation saved in: {save_path}")

def on_load_click():
    file_path = filedialog.askopenfilename(title="Select file: ", filetypes=[("PDF o TXT", "*.pdf *.txt")])
    if file_path:
        load_document_file(file_path)

button_frame = tk.Frame(left_frame)
button_frame.pack(fill=tk.X, padx=5, pady=5)

send_btn = tk.Button(button_frame, text="SEND", command=send_message)
send_btn.pack(side=tk.LEFT, padx=5)

load_btn = tk.Button(button_frame, text="LOAD PDF/TXT", command=on_load_click)
load_btn.pack(side=tk.LEFT, padx=5)

save_btn = tk.Button(button_frame, text="SAVE CONVERSATION", command=save_conversation)
save_btn.pack(side=tk.LEFT, padx=5)

load_conv_btn = tk.Button(button_frame, text="LOAD CONVERSATION", command=load_conversation)
load_conv_btn.pack(side=tk.LEFT, padx=5)

apply_dark_theme = lambda widget: widget.configure(bg="#1e1e1e")
apply_dark_theme(root)

root.mainloop()
