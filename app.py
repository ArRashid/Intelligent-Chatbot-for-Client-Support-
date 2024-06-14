from GinAI.Bot import *
from flask import Flask, jsonify, request, render_template, redirect#sessionsession
from werkzeug.utils import secure_filename
import time
from helper import *


import hashlib
#DOCSEARCH = get_all_pinecone_docsearch()

def calculate_checksum(file_path):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Read the file in binary mode and update the hash object
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)

    # Return the hexadecimal digest of the hash
    return md5_hash.hexdigest()


current_path = os.getcwd()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(
    current_path, "media", "uploaded_files")


@app.route('/')
def home():
    return render_template('index.html')








@app.route('/chatpublic', methods=['POST', 'GET', 'PUT'])
def Chatpublic():
    if request.method == "POST":
        message = request.json['message']
        # Process the user's message and generate a response
        try:
            response = public_chat(message)
            return jsonify({'message': response})
        except Exception as e:
            response = str(e)
            return jsonify({'message': response})

    elif request.method == "GET":
        return render_template("chatpublic.html")

    elif request.method == "PUT":

        return "The PUT CALL SUCCESS"













# clientbot = ClientChatBot(llm=get_llm(),docsearch=DOCSEARCH['CLIENT'])
# clientbot.load_tool_createticket()
# clientbot.load_predifine_tools(tool_names=['human'])
# clientbot.load_tool_multi_pormt_assistance()
# clientbot.agent_chain_with_memory()
# clientmemory =  ConversationBufferMemory()
# clientchat =clientbot.agent_chain_with_memory(memory=clientmemory)
@app.route('/chatclient', methods=['POST', 'GET'])
def Chatclient():
    
    if request.method == "POST":
        message = request.json['message']
        # Process the user's message and generate a response
        try:
            response = clientchat.run(message)
            return jsonify({'message': response})
        except Exception as e:
            response = str(e)
            return jsonify({'message': response})

    elif request.method == "GET":
        return render_template("chatclient.html")
























# teammemory = ConversationBufferMemory()

# teambot = TeamChatBot(docsearch=DOCSEARCH['TEAM'],memory=teammemory,llm=get_llm(temperature=0))
# teambot.define_conv_chain()




@app.route('/chatteam', methods=['POST', 'GET', 'PUT'])
def Chatteam():

    if request.method == "POST":
        message: str = request.json['message']
        chat_session = _get_user_session()
        chatgpt_message = chat_session.get_chatgpt_response(message)
        return jsonify({"message": chatgpt_message})
    
    # print("TEST: initilize object")
    # if request.method == "POST":
    #     print("TEST: USe  oject")
    #     message = request.json['message']
    #     # Process the user's message and generate a response
    #     try:
            
            
    #         response = teambot.query_with_context(message)
    #         return jsonify({'message': response})
    #     except Exception as e:
    #         response = str(e)
    #         return jsonify({'message': response})

    elif request.method == "GET":
        print("TEST: provide interface object")
        return render_template("chatteam.html")



@app.route('/importdata', methods=['POST', 'GET'])
def Importdata():
    if request.method == "POST":
        data = request.json
        import_data(data=data["data"], application_name=data["application_name"], chunk_size=int(
            data["chunk_size"]), chunk_overlap=int(data["chunk_overlap"]))
        response = data
        return render_template('importdata.html')

    elif request.method == "GET":

        return render_template('importdata.html')


@app.route('/xxxx', methods=['POST', 'GET'])
def xxxx():
    if request.method == "POST":
        message = request.json['message']
        time.sleep(5)
        # Process the user's message and generate a response
        response = 'Hello! I received your message: ' + message
        return jsonify({'message': response})

    elif request.method == "GET":
        return render_template('xxxx.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']

        # If the user does not select a file, browser may submit an empty part without filename
        if file.filename == '':
            return 'No file selected'

        # Secure the filename to prevent any malicious behavior
        filename = secure_filename(file.filename)

        # Save the file to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Optionally, you can perform additional processing on the uploaded file here

        return 'File uploaded successfully'















































from GinAI.chat.chat  import ChatSession


from typing import Dict

chat_sessions: Dict[str, ChatSession] = {}

def _get_user_session() -> ChatSession:
    """
    If a ChatSession exists for the current user return it
    Otherwise create a new session, add it into the session.
    """

    chat_session_id = session.get("chat_session_id")
    if chat_session_id:
        chat_session = chat_sessions.get(chat_session_id)
        if not chat_session:
            chat_session = ChatSession()
            chat_sessions[chat_session.session_id] = chat_session
            session["chat_session_id"] = chat_session.session_id
    else:
        chat_session = ChatSession()
        chat_sessions[chat_session.session_id] = chat_session
        session["chat_session_id"] = chat_session.session_id
    return chat_session










if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port="80")
