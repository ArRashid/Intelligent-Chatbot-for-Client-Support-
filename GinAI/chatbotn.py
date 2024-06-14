#from GinAI.default import *
#from GinAI.vectordb import *
import os,json
from typing import Any
from langchain.llms import OpenAI
from langchain import PromptTemplate,LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.tools import StructuredTool
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory,
                                                  ConversationSummaryBufferMemory)
from langchain.callbacks import get_openai_callback
from langchain.agents import initialize_agent ,Tool,AgentType
import tiktoken

from . import *
from GinAI.vectordb import PinconeDb

with open('SETTINGS.json') as json_file:
    SETTINGS = json.load(json_file)['SETTINGS']



def token_len(string: str, encoding_name: str="gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_emmbeding(current_emmbeding=SETTINGS["GLOBAL"]["APPLICATION"]["CURRENT_EMBEDDING"]):
    print("Creating embedding Object where current embeddig is '{0}'.".format(current_emmbeding))
    if current_emmbeding == "OPENAI" :
        embeddings = OpenAIEmbeddings(openai_api_key=SETTINGS["LLM"]["OPENAI"]["API_KEY"])
        return embeddings


def get_llm(llm= "OPENAI"):
    if llm == "OPENAI":
        print("getting OPENAI LLM..")
        os.environ["OPENAI_API_KEY"] = SETTINGS["LLM"]["OPENAI"]["API_KEY"]
        llm_obj = OpenAI()
        return llm_obj

def get_docsearch():
    pincone_obj = PinconeDb()
    print("creating doc search")
    return  pincone_obj.get_docsearch(embedding=get_emmbeding(),pinecone_api_key=SETTINGS['DATABASE']['PINECONE']['PUBLIC']['PINECONE_API_KEY'],pinecone_env=SETTINGS['DATABASE']['PINECONE']['PUBLIC']['PINECONE_ENV'],pinecone_index=SETTINGS['DATABASE']['PINECONE']['PUBLIC']['PINECONE_INDEX'])

# Toools ...............................................

#will delete all tools from the memory 
    def drop_all_tools(self):
        self.tools = []
        print("info : all tools are deleted successfully")



#Tools funsation for creating ticket 
def createticket(subject:str=None,description:str=None):
    ''' when said "create ticket" then exicute this with two inputs: subject(issue) and description(issue detils)."'''

    if subject != None and description != None:
        print(f"#### Ticket has been created with sub :'{subject}' and desc :'{description}' .")
        return f"The ticket has been created succssfully ticket url is : https://ginesys.frshdesk.com/tickets/999999"
        
    else:
        print("Ticket can't be created as the the subject or description is not provied ")
        return "Error"



    



class BaseChatBoot():
    def __init__(self,llm,docsearch,memory=None,chat_preset=None,agent="zero-shot-react-description",verbose=False,token_limitation=SETTINGS['GLOBAL']['CHATS']['TOKEN_LIMITATION']) -> None:
        self.docsearch = docsearch
        self.llm = llm
        self.memory = memory
        self.agent = agent
        self.verbose = verbose
        self.tools = []
        self.chat_preset = chat_preset
        #static defined atribute
        self.max_iterations = 3
        if token_limitation:
            self.token_limitation = SETTINGS['GLOBAL']['CHATS']['TOKEN_LIMITATION']
            self.context_token_limit = SETTINGS['GLOBAL']['CHATS']['CONTEXT_TOKEN_LIMIT']
            self.input_token_limit = SETTINGS['GLOBAL']['CHATS']['INPUT_TOKEN_LIMIT']
            self.history_token_limit = SETTINGS['GLOBAL']['CHATS']['HISTORY_TOKEN_LIMIT']

    def get_context(self,query):
        return self.docsearch.similarity_search(query)

   

    def use_tool_mul(self):
        ctools = StructuredTool.from_function(self.multipmchain)  
        if ctools not in self.tools:
            self.tools.append(ctools)
            print("Mull tamp llm tool added sussfully")
    # def use_tool_create_ticket(self):
    #     ctools = StructuredTool.from_function(createticket)  
    #     if ctools not in self.tools:
    #         self.tools.append(ctools)
    #         print("ticket creation tool added sussfully")
    # def unuse_tool_create_ticket(self):
    #     ctools = StructuredTool.from_function(createticket)  
    #     if ctools not in self.tools:
    #         self.tools.pop(ctools)
    
                


    
    def client_chat_chain(self,template=None,**kwargs):
        agent_chain = initialize_agent(tools=self.tools,llm=self.llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,memory=self.memory,verbose=self.verbose,max_iterations=self.max_iterations,**kwargs)

        return agent_chain


    def agent_with_memory(self,quarry):
        self.agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
        agent_chain = initialize_agent(tools=self.tools,llm=self.llm,agent=self.agent,memory=self.memory,verbose=True,max_iterations=3,)
        result = agent_chain.run(quarry)
        return result

    #this will check memory and call the funsa
    def final_chain(self,input_variable):
        conversation = None
        if self.memory != None:
            conversation = ConversationChain(
            llm=self.llm, memory=self.memory
            )
        else:
            onversation = ConversationChain(
            llm=self.llm
            )
        return conversation.run(input_variable)



    def multipmchain(self,question):
        '''This tool will be used when you have queries related to Ginesys-Funstation information, resolution requirements for related issues, when you need help, or any general queries.'''

        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise and easy to understand manner. \
        When you don't know the answer to a question you admit that you don't know.

        Here is a question:{input}"""


        math_template = """You are a very good mathematician. You are great at answering math questions. \
        You are so good because you are able to break down hard problems into their component parts, \
        answer the component parts, and then put them together to answer the broader question.
        Here is a question:{input}"""

        prompt_infos = [
        {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
        },
        {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
        },
        ]




        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=self.llm,memory=self.memory, output_key="text")







        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template,input_variables=["input"],output_parser=RouterOutputParser(),)
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)

        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,verbose=True
            )

        return chain.run(question)



        




    def check_inputs_tocken(self,user_query:str,):
        context:list = None
        if self.token_limitation :
            if token_len(user_query) > self.input_token_limit :
                return "Provided query is too long , please make it shorter or increare the token_limitation in setting"
            else:
                context = self.docsearch.similasimilarity_search(user_query)
                if token_len(context) > self.context_token_limit : 
                    context = context[:3]
                    if token_len(context) > self.context_token_limit : 
                        context = context[:2]
                        if token_len(context) > self.context_token_limit : 
                            context = context[:1]
                            if token_len(context) > self.context_token_limit : 
                                return "Even a single contex lenth is too long unable to proceed, kindly check context_token_limit in setting"

        self.final_chain()
        


        





    def public_query(self,question,promt_template=None):
        oc = PinconeDb()
        print("getting similar result")
        context = oc.similasimilarity_search(question)[0]
        print("cleating port template ")
        template = """Question: {question}
        
        Context : {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["question","context"])
        #prompt.format(question=question,context=context)
        llm = get_llm()
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain.run({'question': question, 'context': context})






class ClientChatBot(BaseChatBoot):
    def __init__(self, llm, docsearch, memory=None, chat_preset=None, agent="zero-shot-react-description", verbose=False, token_limitation=SETTINGS['GLOBAL']['CHATS']['TOKEN_LIMITATION']) -> None:
        super().__init__(llm, docsearch, memory, chat_preset, agent, verbose, token_limitation)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    
    def init_llm_chain(self,prompt_tamplate):
        chain= LLMChain(llm=self.llm,prompt=prompt_tamplate)
        return chain
    
    def chat(self,question):

        pass

    #This is the main dicision maker agent : who will decide which to need to use for the quarry 
    def decision_maker_agent(self,**kwargs):
        agent_chain = initialize_agent(tools=self.tools,
                                       llm=self.llm,
                                       agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                       verbose=True,
                                       max_iterations=self.max_iterations,
                                       **kwargs)
        return agent_chain
    def marge_context_into_question(self,question):
        context = self.get_context(question)
        tamplate = '''{question} ? \n \n
           
            Relevant document or context: {context} 
        '''.format(context=context,question=question)
        return tamplate
    
    # while decision_maker_agent decided to use this . this is the llm with multiple promt template . it will use the relavent pormt for the task 
    def multi_pormt_assggistance(self,question):
        '''This tool will be used when you have queries related to Ginesys-Funstation information, resolution requirements for related issues, when you need help, or any general queries.'''

        newquestion = self.marge_context_into_question(question)
        print("------------",question)
        ginesys_tech_support_ai_l1 = """'''You are a highly skilled technical support assistant.\
            You excel at answering questions and providing step-by-step solutions .\
            also provide the refarence link or souce information .\
            When you encounter a question you don't know the answer to, you are honest about it.

           
            Here is a question: {input}
            "''"""


       
        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise and easy to understand manner. \
        When you don't know the answer to a question you admit that you don't know.

        Here is a question:{input}"""


        math_template = """You are a very good mathematician. You are great at answering math questions. \
        You are so good because you are able to break down hard problems into their component parts, \
        answer the component parts, and then put them together to answer the broader question.
        Here is a question:{input}"""

        prompt_infos = [
        {
        "name": "Ginesys_support",
        "description": "Good for answering questions about Ginesys application",
        "prompt_template": ginesys_tech_support_ai_l1,
        },
        {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
        },
        ]




        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=self.llm,memory=self.memory, output_key="text")







        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template,input_variables=["input"],output_parser=RouterOutputParser(),)
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)

        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,verbose=True
            )

        return chain.run(newquestion)


    


    #this will laod the multi_pormt_assistance as tool for the dicision maker agent 
    def load_tool_multi_pormt_assistance(self):
        init = StructuredTool.from_function(self.multi_pormt_assistance)  
        if init not in self.tools:
            self.tools.append(init)
            print("info : multi_pormt_assistance tool loaded successfully")
        else:
            print("faild : multi_pormt_assistance tool already loaded")

    

    
    
