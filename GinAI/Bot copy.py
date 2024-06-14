# %%
import json,os
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,StructuredChatAgent,load_tools
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.tools import StructuredTool
from langchain.chains import ConversationChain


from . import *

# %% [markdown]
# loading setting from the setting file

# %%
os.environ["GOOGLE_CSE_ID"] = SETTINGS['TOOLS']['GOOGLE']['CSE_ID']
os.environ["GOOGLE_API_KEY"] = SETTINGS['TOOLS']['GOOGLE']['API_KEY']

# %% [markdown]
# Creating Base Chat bot

# %%
class BaseBot():
    def __init__(self,llm=None,memory=None,tools:list=[],docsearch=None) -> None:
        self.tools:list = tools
        self.memory =  memory
        self.llm = llm
        self.docsearch =docsearch
    def set_llm(self,llm:object)-> None:
        self.llm = llm

        
    def load_predifine_tools(self,tool_names:list,llm=None):
        if not isinstance(tool_names, list):
            tool_names = [tool_names]
        if llm == None:
            llm = self.get_llm()
        self.tools.extend(load_tools(tool_names=tool_names,llm=llm))
        print(f"Info : Tools - {tool_names} are successfully loaded")
    

        
    def get_llm(self,**kwargs):
        os.environ["OPENAI_API_KEY"] = SETTINGS["LLM"]["OPENAI"]["API_KEY"]
        return OpenAI(*kwargs)


    def llm_chain_for_agent(self,llm=None,suffix=None,prefix=None,tools=None):
        if  llm == None:
            llm =self.get_llm()
        if prefix == None:
            prefix = """Have a conversation with a human, answering the following questions as very long article. You have access to the following tools:"""
        if suffix == None:
            suffix = """Begin!"
        

        
            {history}
            Question: {input}
            {agent_scratchpad}"""
        if tools  == None:
            tools = self.tools

        prompt = StructuredChatAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "history", "agent_scratchpad"],)


        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return llm_chain
    

    def agent_chain_with_memory(self,memory=None,tools=None,llm_chain=None):
        #https://python.langchain.com/docs/modules/memory/how_to/agent_with_memory
        if tools == None:
            tools = self.tools
        if llm_chain == None:
            llm_chain = self.llm_chain_for_agent()
        if memory == None:
            memory = self.memory
       
        agent = StructuredChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory,max_iterations=3)
        return agent_chain
    def get_context(self,query):
        if self.docsearch !=  None:
            print("info : getting context from ginesys database")
            context = self.docsearch.similarity_search(query,k=1)
            print(f"details : context =  {context}")
            return context
        else:
            print("Error: The docsearch object is  not provided to the class plase do byusing object.docsearch = 'docsearch object'")
            
    
    
    def marge_context_and_question_as_single_input(self,question):
        context = self.get_context(question)
        tamplate = f'''{question} 
        

                
Context: {context} 
        '''
        print("marged inpu :::::::::::::",tamplate)
        return tamplate
    



# %%
class ClientChatBot(BaseBot):
    def __init__(self, llm=None, memory=None, tools: list =[],docsearch=None) -> None:
        super().__init__(llm, memory, tools,docsearch)


#while decision_maker_agent decided to use this . this is the llm with multiple promt template . it will use the relavent pormt for the task 
    def multi_pormt_assistance(self,question):
        '''This tool will be used when you have queries related to Ginesys-Funstation information, resolution requirements for related issues, when you need help, or any general queries.'''

        the_question_with_contex = self.marge_context_and_question_as_single_input(question=question)
        if self.llm == None:
            self.llm = self.get_llm()
        
        Ginesys_Tech_Support_AI_Template_v1 = """'''You are a highly skilled technical support assistant. \
        You excel at answering questions and providing step-by-step solutions. \
        If the context is available, please provide the reference link or source information. \
        When you encounter a question you don't know the answer to, please indicate "I don't know".
        
        
        Here is a question: {input}
        "''"""



       
        Ginesys_Functional_Support_AI_Template_v1 = """You are knowledgeable about all Ginesys functional uses. \
        You excel at answering questions about how to use and understand Ginesys products. \
        When you encounter a question you don't know the answer to, you are honest about it. \
        If the context is available, please provide the reference link or source information.
        
        
        Here is a question: {input}
        """



        Ginesys_Business_Logic_Support_AI_L1_Template_v2 = """You are highly skilled in business logic and excel at answering business questions. \
        Your strength lies in breaking down complex problems into manageable components, providing answers for each component, and then \
        integrating them to address the broader question.
        
        Here is a question: {input}"""


        prompt_infos = [
        {
        "name": "Ginesys_Tech_Support_AI",
        "description": "Good for answering technical questions, providing step-by-step solutions, and offering reference links or sources when available",
        "prompt_template": Ginesys_Tech_Support_AI_Template_v1,
        },
        {
        "name": "Ginesys_Functional_Support",
        "description": "Good for answering questions about Ginesys functional uses, helping with product usage and understanding.",
        "prompt_template": Ginesys_Functional_Support_AI_Template_v1,
        },
        {
        "name": "Ginesys_Business_Support",
        "description": " Good for answering business questions, breaking down complex problems, and providing integrated solutions based on components.",
        "prompt_template": Ginesys_Business_Logic_Support_AI_L1_Template_v2,
        },
        ]




        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=self.llm,output_key="text")







        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template,input_variables=["input"],output_parser=RouterOutputParser(),)
        router_chain = LLMRouterChain.from_llm(llm=self.llm, prompt=router_prompt)

        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,verbose=True
            )
        

        if self.docsearch != None:
            return chain.run(self.marge_context_and_question_as_single_input(question=question))
        else:
            return chain.run(question)


            
    def createticket(self,subject:str=None,description:str=None):
        '''when the user is not satifyed with answer and asked for create a ticke and there is no ticket already created then use this .this  requires two inputs: subject (issue) and description (issue details)."'''

        if subject != None and description != None:
            print(f"#### Ticket has been created with sub :'{subject}' and desc :'{description}' .")
            return f"The ticket has been created succssfully ticket url is : https://ginesys.frshdesk.com/tickets/999999"
        
        else:
            print("Ticket can't be created as the the subject or description is not provied ")
            return "Error"
    #this will laod the multi_pormt_assistance as tool for the dicision maker agent 
    def load_tool_multi_pormt_assistance(self):
        init = StructuredTool.from_function(self.multi_pormt_assistance)  
        if init not in self.tools:
            self.tools.append(init)
            print("info : multi_pormt_assistance tool loaded successfully")
        else:
            print("faild : multi_pormt_assistance tool already loaded")
    def load_tool_createticket(self):
        init = StructuredTool.from_function(self.createticket)  
        if init not in self.tools:
            self.tools.append(init)
            print("info : Create Ticket tool loaded successfully")
        else:
            print("faild : Create Ticket tool already loaded")
    



class TeamChatBot(BaseBot):
    def __init__(self, llm:object=None, memory=None, tools: list = [], docsearch=None) -> None:
        super().__init__(llm, memory, tools, docsearch)

    def multi_pormt_assistance(self,question):
        '''This tool will be used when you need help with any issues, errors, or information related to the application, its functionality, documentation, or any reference issues.'''
        

        
        Error_Details_Template_v1 = """Answer the question based on the context below.\
Provide a long details answer with formate of technical note .\
If the question cannot be answered using the information provided answer
with "I don't know : As The Ginesys AI is under devolopment ".

Question: {input}

Answer: """
        How_To_Guid_Template_v1 = """Answer the question based on the context below.\
Provide long and full answer with formate of 'How to Guide note' .\
Also provide  soure path from context for refarence.\
If the question cannot be answered using the information provided answer
with "I don't know : As The Ginesys AI is under devolopment ".

Question: {input}

Answer: """
        Info_Template_v1 = """Answer the question based on the context below.\
Provide long and full answer with formate of 'information note' .\
Also provide  soure path from context for refarence.\
If the question cannot be answered using the information provided answer
with "I don't know : As The Ginesys AI is under devolopment ".

Question: {input}

Answer: """
     




        prompt_infos = [
        {
        "name": "Error_Details",
        "description": "Great for providing detailed explanations, resolutions, and references for errors.",
        "prompt_template": Error_Details_Template_v1,
        },
        {
        "name": "How_Guidence",
        "description": "Good for answering questions which like how to or how .",
        "prompt_template": How_To_Guid_Template_v1,
        },
         {
        "name": "How_Guidence",
        "description": "Good for answering questions which related information releted .",
        "prompt_template": Info_Template_v1,
        },
       
        ]




        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain
        default_chain = ConversationChain(llm=self.llm,output_key="text")







        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template,input_variables=["input"],output_parser=RouterOutputParser(),)
        router_chain = LLMRouterChain.from_llm(llm=self.llm, prompt=router_prompt)

        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,verbose=True
            )
        
        if self.docsearch != None or True:
            newquestion =self.marge_context_and_question_as_single_input(question=question)
            print(newquestion)
            return chain.run(input=newquestion)
        else:
            return chain.run(question)
    
    

       #this will laod the multi_pormt_assistance as tool for the dicision maker agent 
    def load_tool_multi_pormt_assistance(self):
        init = StructuredTool.from_function(self.multi_pormt_assistance)  
        if init not in self.tools:
            self.tools.append(init)
            print("info : multi_pormt_assistance tool loaded successfully")
        else:
            print("faild : multi_pormt_assistance tool already loaded")
    def load_tool_query_context(self):
        self.define_conv_chain()
        init = StructuredTool.from_function(self.query_with_context)  
        if init not in self.tools:
            self.tools.append(init)
            print("info : multi_pormt_assistance tool loaded successfully")
        else:
            print("faild : multi_pormt_assistance tool already loaded")
    def define_conv_chain(self,memory=None,llm=None,prompt=None,**kwagrs):
        if llm == None:
            llm = get_llm()
        if prompt == None:
            template = """Answer the question based on the context below.\
Provide full details answer with proper formated profesonaly will all the context.\
Also provide soure path from context for more info.\
If the question cannot be answered using the information provided answer
with "I don't know".

Question: {query}

Context: {context}

Answer: """

            prompt_template = PromptTemplate(
    input_variables=["query",'context'],template=template)
            template= '''
            '''
            prompt = prompt_template
        
        conversation_buf=LLMChain(llm=self.llm,prompt=prompt,**kwagrs)
            
            
        
        self.current_chain =conversation_buf
        
        return conversation_buf
    def query_with_context(self,query):
        '''This tool will be used when you need help with any issues, errors, or information related to the application, its functionality, documentation, or any reference issues.'''
        

        context = self.get_context(query)
        return self.current_chain.run({'query':query,'context':context})
