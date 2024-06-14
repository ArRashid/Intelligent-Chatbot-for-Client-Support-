import json,os
import tiktoken
from .vectordb import PinconeDb
from langchain.llms import OpenAI
with open('SETTINGS.json') as json_file:
    SETTINGS = json.load(json_file)['SETTINGS']



def token_len(string: str, encoding_name: str="gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

from langchain.embeddings import OpenAIEmbeddings
def get_emmbeding(current_emmbeding=SETTINGS["GLOBAL"]["APPLICATION"]["CURRENT_EMBEDDING"]):
    print("info : Creating embedding Object where current embeddig is '{0}'.".format(
        current_emmbeding))
    if current_emmbeding == "OPENAI":         
        embeddings = OpenAIEmbeddings(openai_api_key=SETTINGS["LLM"]["OPENAI"]["API_KEY"])
        return embeddings
    
def get_llm(current_llm=SETTINGS["GLOBAL"]["APPLICATION"]["CURRENT_LLM"],**kwargs) -> object:
    if current_llm == "OPENAI":
        print("info :Creating LLM Object where current LLM is '{0}'.".format(
            current_llm))
        os.environ["OPENAI_API_KEY"] = SETTINGS["LLM"]["OPENAI"]["API_KEY"]
        llm_obj = OpenAI(**kwargs)
        return llm_obj
    
def get_all_pinecone_docsearch():
    print("Creating pinecone docsearch Object for CLIENT, PUBLIC and TEAM.")
    pincone_obj = PinconeDb()
    emedding=get_emmbeding()
    client_docsearch = pincone_obj.get_docsearch(embedding=emedding, pinecone_api_key=SETTINGS['DATABASE']['PINECONE']['CLIENT']['PINECONE_API_KEY'], pinecone_env=SETTINGS[
                                                 'DATABASE']['PINECONE']['CLIENT']['PINECONE_ENV'], pinecone_index=SETTINGS['DATABASE']['PINECONE']['CLIENT']['PINECONE_INDEX'])
    public_docsearch = pincone_obj.get_docsearch(embedding=emedding, pinecone_api_key=SETTINGS['DATABASE']['PINECONE']['PUBLIC']['PINECONE_API_KEY'], pinecone_env=SETTINGS[
                                                 'DATABASE']['PINECONE']['PUBLIC']['PINECONE_ENV'], pinecone_index=SETTINGS['DATABASE']['PINECONE']['PUBLIC']['PINECONE_INDEX'])
    team_docsearch = pincone_obj.get_docsearch(embedding=emedding, pinecone_api_key=SETTINGS['DATABASE']['PINECONE']['TEAM']['PINECONE_API_KEY'], pinecone_env=SETTINGS[
                                               'DATABASE']['PINECONE']['TEAM']['PINECONE_ENV'], pinecone_index=SETTINGS['DATABASE']['PINECONE']['TEAM']['PINECONE_INDEX'])

    return {"CLIENT": client_docsearch, "PUBLIC": public_docsearch, "TEAM": team_docsearch}


#EXPERIMETAL ===========================================================================================
import pickle

def get_database():
    try:
        with open('database.bin', 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    except:
        data = []
        with open('database.bin', 'wb') as file:
            pickle.dump(data, file)
        return data


def set_database(data, old_data=get_database()):
    old_data.append(data)
    with open('data.bin', 'wb') as file:
        pickle.dump(old_data, file)




print("info : madule : GinAI madule is imported successfully")