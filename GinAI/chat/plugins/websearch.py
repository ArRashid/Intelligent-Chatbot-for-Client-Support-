from .plugin import PluginInterface,SETTINGS
from typing import  Dict
import requests
import os


GOOGLE_CSE_ID = SETTINGS['TOOLS']['GOOGLE']['CSE_ID']
GOOGLE_API_KEY = SETTINGS['TOOLS']['GOOGLE']['API_KEY']
BASE_URL = "https://www.googleapis.com/customsearch/v1"

class WebSearchPlugin(PluginInterface):
    def get_name(self) -> str:
        """
        return the name of the plugin (should be snake case)
        """
        return "websearch"
    
    def get_description(self) -> str:
        return """
        Executes a web search for the given query
        and returns a list of snipptets of matching
        text from top 10 pages
        """
    

    def get_parameters(self) -> Dict:
        """
        Return the list of parameters to execute this plugin in the form of
        JSON schema as specified in the OpenAI documentation:
        https://platform.openai.com/docs/api-reference/chat/create#chat/create-parameters
        """
        parameters = {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "the user query"
                }
            }
        }
        return parameters
    
    def execute(self, **kwargs) -> Dict:
        """
        Execute the plugin and return a JSON response.
        The parameters are passed in the form of kwargs
        """

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY
        }

        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": kwargs["q"]
        }

        response =  response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            results = response.json()['web']['results']
            snippets = [r['description'] for r in results]
            return {"web_search_results": snippets}
        else:
            return {"error":
                    f"Request failed with status code: {response.status_code}"}