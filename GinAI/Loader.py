#!/usr/bin/env python3
import os,json
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import pinecone
from langchain.vectorstores import Pinecone
from langchain.document_loaders import WebBaseLoader,PyPDFLoader,UnstructuredPDFLoader,CSVLoader,ConfluenceLoader
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    #UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings


from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from GinAI.TicketAPI import get_ticket_convasation

class StringLoader(BaseLoader):
    """Load text files."""

    def __init__(self, string: str,source:str=None):
        """Initialize text"""
        self.string = string
        if source == None:
            self.source ="Directly imported as string or no source provided"
        else:
            self.source =source
    
    def load(self) -> List[Document]:
        """Load from string."""
        metadata = {"source": self.source}
        return [Document(page_content=self.string, metadata=metadata)]
    



with open('SETTINGS.json') as json_file:
    SETTINGS = json.load(json_file)['SETTINGS']


def get_embeding(current_emmbeding=SETTINGS["GLOBAL"]["APPLICATION"]["CURRENT_EMBEDDING"]):
    print("Creating embedding Object where current embeddig is '{0}'.".format(current_emmbeding))
    if current_emmbeding == "OPENAI" :
        embeddings = OpenAIEmbeddings(openai_api_key=SETTINGS["LLM"]["OPENAI"]["API_KEY"])
        return embeddings

default_embedding = get_embeding()
default_chunk_size  = SETTINGS["GLOBAL"]["APPLICATION"]["CHUNK_SIZE"]
default_chunk_overlap  = SETTINGS["GLOBAL"]["APPLICATION"]["CHUNK_OVERLAP"]



default_database_credentials = SETTINGS['DATABASE']['PINECONE']['PUBLIC']


print(default_database_credentials)

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    #".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def text_spilt(documents,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def process_documents(source_directory,ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    return documents

#===============Custome Fun for help the main method =====================================

def convert_splited_texts_into_list(splited_texts=None):
        list_of_text = []
        for text in splited_texts :
            list_of_text.append(str(text))
        return list_of_text


def load_documents_from_links(links:list[str],ignored_files: List[str] = []) -> List[Document]:
    if not isinstance(links, list):
        links = [links]
    print("starting loading total {0} web page  data ....".format(len(links)))
   
    filtered_links = [link for link in links if link not in ignored_files]
    loader = WebBaseLoader(filtered_links)
    loader.requests_per_second = 1
    documents = loader.load()
    print(f"The document has loded count of documents is {len(documents)} pcs .")
    return documents


def load_document_from_confluence(space_key,limit=50,url="",username="",api_key=""):
   loader = ConfluenceLoader(
        url=url,#"https://yoursite.atlassian.com/wiki",
        username=username,
        api_key=api_key
        )
   documents = loader.load(space_key=space_key,limit=50)
   return documents

def load_document_from_text(file_path: str,encoding= "utf8") -> List[Document]:
   loader = TextLoader(file_path,encoding= encoding)
   documents = loader.load()
   return documents

def load_document_from_texts(file_paths: list[str],encoding= "utf8") -> List[Document]:
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path,encoding= encoding)
        doc =loader.load()
        documents.extend(doc) 
    return documents

def load_document_from_string(string: str,source:str=None) -> List[Document]:
   loader = StringLoader(string=string,source=source)
   documents = loader.load()
   return documents


def load_documents_from_pdf(paths=None):
        if not isinstance(paths, list):
            paths = [paths]
        documents = []
        if len(paths) >= 1 :
            for path in paths :
                loader = PyMuPDFLoader(path)
                new_documents = loader.load_and_split()
                documents.extend(new_documents)
            return documents
        else:
             print("Invalid oparation : The PDF file is less than One")

def load_documents_from_pdf_perpage(paths=None):
        if not isinstance(paths, list):
            paths = [paths]
        documents = []
        if len(paths) >= 1 :
            for path in paths :
                loader = PyPDFLoader(path)
                new_documents = loader.load_and_split()
                documents.extend(new_documents)
            return documents
        else:
             print("Invalid oparation : The PDF file is less than One")

def load_documents_from_pdf_unstructured(paths=None):
    if not isinstance(paths, list):
            paths = [paths]
    documents = []
    if len(paths) >= 1 :
        for path in paths :
            loader = UnstructuredPDFLoader()
            new_documents = loader.load(path,mode="elements")
            documents.extend(new_documents)
        return documents
    else: 
        print("Invalid oparation : The PDF file is less than One")



#database loader

def load_data_into_pinconedb(list_of_texts,embeddings,database_credentials:dict):
    try:
        pinecone.init(
        api_key=database_credentials["PINECONE_API_KEY"],  # find at app.pinecone.io
        environment=database_credentials["PINECONE_ENV"]    )
        if not database_credentials["PINECONE_INDEX"] in pinecone.list_indexes():
        	    pinecone.create_index(database_credentials["PINECONE_INDEX"], dimension=12)
        docsearch = Pinecone.from_texts(list_of_texts, embeddings, index_name=database_credentials["PINECONE_INDEX"])
        return docsearch
    except Exception as e:
        print(f"Error While data importing Error: {e}")







#main funsations are in below
def load_directory(source_directory,embedding=default_embedding,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap,database_credentials=default_database_credentials):
    #
    # does_vectorstore_exist("test")
    # Update and store vectorstore
    print(f"Loading files from dir : {source_directory}. May take some minutes...")
    docs = process_documents(source_directory=source_directory,ignored_files=["ignored.fils"])
    texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    print(f"Appending to from {source_directory} to pincone vectorstore at index :")
    load_data_into_pinconedb(texts,embeddings=embedding,database_credentials=database_credentials)
    print(f"Ingestion complete! You can now run chats to query your documents")

def load_weblinks(links,embedding=default_embedding,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap,database_credentials=default_database_credentials):
    print(f"Loading links from dir : {links}. May take some minutes...")
    docs = load_documents_from_links(links=links)
    texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    print(f"Appending to from link's doc to pincone vectorstore at index : ")
    load_data_into_pinconedb(texts,embeddings=embedding,database_credentials=database_credentials)
    print(f"Ingestion complete! You can now run chats to query your documents")



def load_pdf(paths,pdftype=None,embedding=default_embedding,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap,database_credentials=default_database_credentials):
    print(f"Loading pdf documet , total pdf path count : {len(paths)}....")
    if isinstance(paths, str):
        paths = [paths]
    print("XXXXXXXXXXXXXXXXXXXx",paths)
    if pdftype == None:
        docs = load_documents_from_pdf(paths=paths)
    elif pdftype.upper() == 'PP' :
        docs = load_documents_from_pdf_perpage(paths=paths)
    elif pdftype.upper() == "US":
        docs = load_documents_from_pdf_unstructured(paths=paths)
    print(type(docs),len(docs))
    texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    print(f"Appending to from pdfs file to pincone vectorstore at index")
    load_data_into_pinconedb(texts,embeddings=embedding,database_credentials=database_credentials)
    print(f"Ingestion complete! You can now run chats to query your documents")

    
def load_confluence(space_key,limit=50,embedding=default_embedding,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap,database_credentials=default_database_credentials):
    print(f"Loading docs from confluence space: {space_key}. May take some minutes...")
    docs = load_document_from_confluence(space_key=space_key,limit=limit)
    texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    list_of_string = convert_splited_texts_into_list(texts)
    print(f"Appending to from confluence to pincone vectorstore at index ")
    load_data_into_pinconedb(list_of_string,embeddings=embedding,database_credentials=database_credentials)
    print(f"Ingestion complete! You can now run chats to query your documents")

def load_string(string,embedding=default_embedding,chunk_size=2000,chunk_overlap=50,database_credentials=default_database_credentials):
    print(f"Loading docs from provided string, lenth of the string is: {len(string)}...")
    docs = load_document_from_string(string=string)
    texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    list_of_string = convert_splited_texts_into_list(texts)
    print(f"Appending data from the string into pincone vectorstore at index ")
    load_data_into_pinconedb(list_of_texts=list_of_string,embeddings=embedding,database_credentials=database_credentials)
    print(f"Ingestion complete! You can now run chats to query your documents")

def load_text(paths,embedding=default_embedding,chunk_size=500,chunk_overlap=50,database_credentials=default_database_credentials):
    print(f"Loading texts : {paths}. May take some minutes...")
    if isinstance(paths, str):
        paths = [paths]
    docs = load_document_from_texts(file_paths=paths)
    print(f"Texts is loaded as document count of : {len(docs)}....")
    texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    list_of_string = convert_splited_texts_into_list(texts)
    print(f"Appending to from texts to pincone vectorstore at index ")
    load_data_into_pinconedb(list_of_texts=list_of_string,embeddings=embedding,database_credentials=database_credentials)
    print(f"Ingestion complete! You can now run chats to query your documents")

def load_ticket(ticket,embedding=default_embedding,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap,database_credentials=default_database_credentials):
    print(f"Loading ticket convasation from ticket id : {ticket}...")
    string,source,status = get_ticket_convasation(ticketid=ticket)
    if status == 200:
        docs = load_document_from_string(string=string,source=source)
        texts = text_spilt(docs,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        list_of_string = convert_splited_texts_into_list(texts)
        print(f"Appending data from the string into pincone vectorstore at index ")
        load_data_into_pinconedb(list_of_texts=list_of_string,embeddings=embedding,database_credentials=database_credentials)
        print(f"Ingestion complete! You can now run chats to query your documents")

def load_tickets(starting=123473,ending=124456):
    while starting != ending :
        load_ticket(starting)
        starting += 1

if __name__ == "__main__":
  
    print("start testing ")
    load_tickets()
    #doneload_string(string=st)
    #done,load_weblinks(links="https://www.ginesys.in")
    #load_confluence(space_key="PUB",chunk_size=450,chunk_overlap=50)
    #done #load_text(paths="C:/Users/arras/Desktop/test_import/Ginesys.txt",chunk_size=10,chunk_overlap=0)
    #doneload_pdf(paths="C:/Users/arras/Desktop/test_import/Ginesys.pdf",chunk_size=10,chunk_overlap=0)
    #done#load_directory(source_directory="C:/Users/arras/Desktop/test_import")
    print("########The has been loaded ")