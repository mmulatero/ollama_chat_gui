# ollama_chat_gui
OLLAMA CHAT GUI

# Prerequisite:

Python 3.13.3

pip install langchain_ollama
pip install PyPDF2
pip install langchain-huggingface
pip install frontend
pip install tools

# SETUP:

1) Download your desired models from here: https://ollama.com/search
    i.e.: deepseek-r1:8b    <-- This could be good on a >= 10GB VRAM GPU
          gemma3:12b        <-- This could be good on a >= 16GB VRAM GPU

2) After the model download execute once the model with ollama run command to download the docker file:
    ollama run deepseek-r1:8b

3) You can check the installed models with the command: 
    ollama list

4) Update the model name in the 'ollama_chat_gui.py' source variable reference:
    MODEL_NAME = "deepseek-r1:8b"

5) Setup other variables for desired behavior:
    TEMPERATURE = 0.6       # lower temperature for more precise response vs higher for more creative response
    CTX_WINDOW = 16384	    # CONTEXT WINDOW -> min 2048 - max 131072 (more context windows requires more VRAM!!!)
    KEEP_ALIVE = 0

# RUN:

python ollama_chat_gui.py

NOTE: the application startup will take a while silently and loading time depends on the running machine, so please wait...
