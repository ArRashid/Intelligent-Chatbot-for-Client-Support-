import os
from langchain.llms import OpenAI
import json
from GinAI.Loader import load_confluence, load_directory, load_pdf, load_string, load_text, load_weblinks
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import ConfluenceLoader
from GinAI import *



upload_path = os.path.join(os.getcwd(), "media", "uploaded_files")
# DATA LOADER OBJECTS -----------------------------------------------------------------
def add_https(links: dict):
    result = []
    for link in links:
        if link.startswith("http://") or link.startswith("https://"):
            result.append(link)
        else:
            result.append("https://" + link)
    return result


def add_uloaded_path(paths: dict):
    result = []
    for path in paths:
        result.append(os.path.join(upload_path, path))
    return result


def separate_paths_by_type(data):
    paths_by_type = {}

    for item in data:
        item_type = item['type']
        item_path = item['path']

        if item_type in paths_by_type:
            paths_by_type[item_type].append(item_path)
        else:
            paths_by_type[item_type] = [item_path]

    return paths_by_type


def is_link_not_imported():
    if True:
        return True
    else:
        print("the link is alred imported into the database")


# def single_import(import_type,path,embedding,chunk_size,chunk_overlap,pinecone_api_key,pinecone_env,pinecone_index):
#     if import_type == "LINK":
#         link = add_https(path)
#         if is_link_not_imported ():
#             try:
#                 docs=Myloder.load_documents_from_links(links=[link])
#                 SpText =Myloder.create_splited_text(documents=docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#                 #Myloder.import_into_db()
#                 Myloder.load_data_into_pinconedb(Myloder.convert_splited_texts_into_list(SpText),embeddings=embedding,pinecone_api_key=pinecone_api_key,pinecone_env=pinecone_env,pinecone_index=pinecone_index)
#                 print("Successfully imported LINK : {0} ".format(link))
#             except  Exception as e:
#                 print("Faild to import the LINK : {0} # THE ERROR IS : {1}".format(link,e))

#     elif import_type == "PDF":
#         full_path = os.path.join(upload_path,path)
#         docs=Myloder.load_documents_from_pdf_perpage([full_path])
#         SpText =Myloder.create_splited_text(documents=docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#         #Myloder.convert_splited_texts_into_list()
#         #Myloder.import_into_db()
#         Myloder.load_data_into_pinconedb(Myloder.convert_splited_texts_into_list(SpText),embeddings=embedding,pinecone_api_key=pinecone_api_key,pinecone_env=pinecone_env,pinecone_index=pinecone_index)
#         print("Successfully imported PDF : {path} ")
#     elif import_type == "CSV":
#         full_path = os.path.join(upload_path, path)
#         docs=Myloder.load_documents_from_csv([full_path])
#         SpText =Myloder.create_splited_text(documents=docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#         #Myloder.convert_splited_texts_into_list()
#         #Myloder.import_into_db()
#         Myloder.load_data_into_pinconedb(Myloder.convert_splited_texts_into_list(SpText),embeddings=embedding,pinecone_api_key=pinecone_api_key,pinecone_env=pinecone_env,pinecone_index=pinecone_index)
#         print("Successfully imported CSV : {path} ")
#     else:
#         print("The import_type is not valid : where type is {0} and path is {1}".format(import_type,path))


def import_data(data: list, application_name: str, chunk_size: int, chunk_overlap: int, mode: str = "single"):
    if application_name == "Team":
        pinecone_db = SETTINGS['DATABASE']['PINECONE']['TEAM']
    elif application_name == "Client":
        pinecone_db = SETTINGS['DATABASE']['PINECONE']['CLIENT']
    elif application_name == "Public":
        pinecone_db = SETTINGS['DATABASE']['PINECONE']['PUBLIC']
    data = separate_paths_by_type(data=data)
    dataase_cred = SETTINGS['DATABASE']['PINECONE'][str(
        application_name.upper())]
    if 'LINK' in data:
        try:
            list_link = add_https(data['LINK'])
            load_weblinks(links=list_link, chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap, database_credentials=dataase_cred)
        except Exception as e:
            print("Link  import faild due to Error : ", e)
    if 'PDF' in data:
        try:
            list_paths = add_uloaded_path(data['PDF'])
            load_pdf(paths=list_paths, chunk_size=chunk_size,
                     chunk_overlap=chunk_overlap, database_credentials=dataase_cred)
        except Exception as e:
            print("Text files  import faild due to Error : ", e)
    if 'TXT' in data:
        try:
            list_paths = add_uloaded_path(data['TXT'])
            print(list_paths)
            load_text(paths=list_paths, chunk_size=chunk_size,
                      chunk_overlap=chunk_overlap, database_credentials=dataase_cred)
        except Exception as e:
            print("Text files  import faild due to Error : ", e)


# Chat related ojects ----------------------------------------------------------------------------------------------------------------


# def public_chat(query, agent=None, mode=None, context_count=4, template=templates[0]):
#     context = docsearch["PUBLIC"].similarity_search(query)
#     print(len(context[0]), len(context[1]), len(context[2]), len(context[3]))
#     prompt = PromptTemplate(template=template, input_variables=[
#                             "question", "context"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     return llm_chain.run({'question': query, 'context': context})


# def client_chat(query, agent=None, mode=None, context_count=4, template=templates[0]):
#     context = docsearch["PUBLIC"].similarity_search(query)
#     print(len(context[0]), len(context[1]), len(context[2]), len(context[3]))
#     prompt = PromptTemplate(template=template, input_variables=[
#                             "question", "context"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     return llm_chain.run({'question': query, 'context': context})


# def team_chat(query, agent=None, mode=None, context_count=4, template=templates[0]):
#     context = docsearch["PUBLIC"].similarity_search(query)
#     print(len(context[0]), len(context[1]), len(context[2]), len(context[3]))
#     prompt = PromptTemplate(template=template, input_variables=[
#                             "question", "context"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     return llm_chain.run({'question': query, 'context': context})


# # for importing data form backend
# def my_customloder(docs, chunk_size, chunk_overlap, embedding=get_emmbeding(), pinecone_api_key=SETTINGS['DATABASE']['PINECONE']['CLIENT']["PINECONE_API_KEY"], pinecone_env=SETTINGS['DATABASE']['PINECONE']['CLIENT']["PINECONE_ENV"], pinecone_index=SETTINGS['DATABASE']['PINECONE']['CLIENT']["PINECONE_INDEX"]):
#     SpText = Myloder.create_splited_text(
#         documents=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     # Myloder.import_into_db()
#     Myloder.load_data_into_pinconedb(Myloder.convert_splited_texts_into_list(
#         SpText), embeddings=embedding, pinecone_api_key=pinecone_api_key, pinecone_env=pinecone_env, pinecone_index=pinecone_index)
#     print("Successfully imported total documents count is : {0} ".format(docs))
