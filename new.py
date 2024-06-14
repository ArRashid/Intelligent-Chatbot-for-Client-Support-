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

# %%
with open('SETTINGS.json') as json_file:
    SETTINGS = json.load(json_file)['SETTINGS']

# %%
class BaseBot():
    def __init__(self,memory=None,tools:list=[]) -> None:
        self.tools:list = tools
        self.memory =  memory
    def load_predifine_tools(self,tool_names:list,llm=None):
        if llm == None:
            llm = self.llm()
        self.tools.extend(load_tools(tool_names=tool_names,llm=llm))
    

        
    def llm(self):
        os.environ["OPENAI_API_KEY"] = SETTINGS["LLM"]["OPENAI"]["API_KEY"]
        return OpenAI(temperature=0)


    def llm_chain(self,llm=None,suffix=None,prefix=None,tools=None):
        if  llm == None:
            llm =self.llm()
        if prefix == None:
            prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
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
            llm_chain = self.llm_chain()
        if memory == None:
            memory = self.memory
       
        agent = StructuredChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory)
        return agent_chain
    







# while decision_maker_agent decided to use this . this is the llm with multiple promt template . it will use the relavent pormt for the task 
    def multi_pormt_assistance(self,question):
        '''This tool will be used when you have queries related to Ginesys-Funstation information, resolution requirements for related issues, when you need help, or any general queries.'''

        newquestion = question
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
        

        return "Soory Ginesys is busy"
    #this will laod the multi_pormt_assistance as tool for the dicision maker agent 
    def load_tool_multi_pormt_assistance(self):
        init = StructuredTool.from_function(self.multi_pormt_assistance)  
        if init not in self.tools:
            self.tools.append(init)
            print("info : multi_pormt_assistance tool loaded successfully")
        else:
            print("faild : multi_pormt_assistance tool already loaded")



# %%
memory = ConversationBufferMemory(input_key="history")
bot = BaseBot(memory=memory)
bot.load_tool_multi_pormt_assistance()
chain = bot.agent_chain_with_memory()

chain.run("what is Gineys?")


