#https://codeconfessions.substack.com/p/creating-chatgpt-plugins-using-the

import openai
import requests
import json
from typing import List, Dict
import uuid

class Conversation:
    """
    This class represents a conversation with the ChatGPT model.
    It stores the conversation history in the form of a list of
    messages.
    """
    def __init__(self):
        self.conversation_history: List[Dict] = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)


GPT_MODEL = "gpt-3.5-turbo-0613"
SYSTEM_PROMPT = """
    You are a helpful AI assistant. You answer the user's queries.
    NEVER make up an answer.
    If you don't know the answer,
    just respond with "I don't know".
"""

class ChatSession:
    """
    Represents a chat session.
    Each session has a unique id to associate it with the user.
    It holds the conversation history
    and provides functionality to get new response from ChatGPT
    for user query.
    """
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation = Conversation()
        self.conversation.add_message("system", SYSTEM_PROMPT)
    
    def get_messages(self) -> List[Dict]:
        """
        Return the list of messages from the current conversation
        """
        # Exclude the SYSTEM_PROMPT when returning the history
        if len(self.conversation.conversation_history) == 1:
            return []
        return self.conversation.conversation_history[1:]
    
    def get_chatgpt_response(self, user_message: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        self.conversation.add_message("user", user_message)
        try:
            chatgpt_response = self._chat_completion_request(
                self.conversation.conversation_history
                )
            chatgpt_message = chatgpt_response.get("content")
            self.conversation.add_message("assistant", chatgpt_message)
            return chatgpt_message
        except Exception as e:
            print(e)
            return "something went wrong"

    def _chat_completion_request(self, messages: List[Dict]):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        json_data = {"model": GPT_MODEL,
                     "messages": messages,
                     "temperature": 0.7}
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            return response.json()["choices"][0]["message"]
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
