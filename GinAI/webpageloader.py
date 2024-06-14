#from dotenv import load_dotenv
from langchain import HuggingFaceHub 
import pinecone
from langchain.vectorstores import Pinecone

#Global variable 
PINECONE_API_KEY = "ce048786-2c99-414f-bf1c-28fa8f66b406"
PINECONE_ENV = "us-west4-gcp-free"
#Global Default variable
default_index_name = "gin-ai-db"
default_input_file = 'Links.txt'
default_output_file = 'out.txt'
default_chunk_size = 2500
default_chunk_overlap = 100

#import os
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#load_dotenv()

def read_file_by_line(filename=default_input_file):
    print("reading the txt files links .....")
    lines = []  # List to store the lines

    # Open the file in read mode
    with open(filename, 'r') as file:
    # Read each line and store it in the list
        for line in file:
            lines.append(line.strip())  # Use strip() to remove the newline character
    return lines


def save_text_to_file(text, filename=default_output_file):
    print("saving output in a file .....")
    try:
        with open(filename, 'w') as file:
            file.write(text)
        print("Text saved to", filename)
    except IOError:
        print("Error: Unable to save text to", filename)


def laoding_data(links):
    print("starting loading data ....")
    from langchain.document_loaders import WebBaseLoader
    loader = WebBaseLoader(links)
    loader.requests_per_second = 1
    docs = loader.load()
    print(f"The document has loded count of documents is {len(docs)} pcs .")
    return docs

def split_text(documents,chunk_size=default_chunk_size,chunk_overlap=default_chunk_overlap):
    print("starting tex spliting ....")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def get_embeddings():
    print("Creating embeddings .....")
    from langchain.embeddings import HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'} # cuda if you have GPU

    embeddings_out = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs
    )
    return embeddings_out




def import_pinecone_db(list_of_texts,embeddings,index_name = default_index_name):
    print("Starting import into pinecone db ....")
    # import pinecone
    # from langchain.vectorstores import Pinecone
    # PINECONE_API_KEY = "ce048786-2c99-414f-bf1c-28fa8f66b406"
    # PINECONE_ENV = "us-west4-gcp-free"

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )

    

    #docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    docsearch = Pinecone.from_texts(list_of_texts, embeddings, index_name=index_name)

    # if you already have an index, you can load it like this
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # query = "What did the president say about Ketanji Brown Jackson"
    # res = docsearch.similarity_search(query)
    # print(res)
    return docsearch



def get_pinecone_db(embeddings,index_name = default_index_name):
    print("Getting vectordb ....")
    # import pinecone
    # from langchain.vectorstores import Pinecone
    # PINECONE_API_KEY = "ce048786-2c99-414f-bf1c-28fa8f66b406"
    # PINECONE_ENV = "us-west4-gcp-free"

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )

    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch




def docs_to_str(texts):
    #TO FIX BELOW ERROR ADDED THIS CUSTOME CODE FOR CONVERT THE OBJECT FROM lanchain document(<class 'langchain.schema.Document'>) to text
    # """
    #     /File "C:\Users\abdur.m\AppData\Local\Programs\Python\Python311\Lib\site-packages\langchain\embeddings\huggingface.py", line 72, in <lambda>
    #     texts = list(map(lambda x: x.replace("\n", " "), texts))
    #                            ^^^^^^^^^
    #     AttributeError: 'Document' object has no attribute 'replace' . """
    print("the text is converting....")
    list_of_text = []
    for text in texts :
        text_new = str(text)
        list_of_text.append(text_new)

    return list_of_text



def load_by_chunks(list_of_links=read_file_by_line(),incremental=4,start=0):
    embeddings = get_embeddings()
    total_links =  len(list_of_links)
    startindex = start
    endindex = None
    if total_links > incremental:
        endindex = start + incremental
    else:
        endindex = start + total_links

    while total_links > 0 :
        print(f"Loading data current start : {startindex} end : {endindex}")
        que = list_of_links[startindex:endindex]
        docload = laoding_data(que)
        texts = split_text(docload)
        new_text = docs_to_str(texts)
        docsearch = import_pinecone_db(new_text,embeddings)
        if total_links > incremental:
            endindex = endindex + incremental
            startindex = startindex + incremental
            total_links = total_links - incremental
        else:
            endindex = endindex + total_links
            startindex = startindex + incremental
            total_links = total_links - incremental
        
    print("all links are done completed ...")







def load_all():
    docs = laoding_data(read_file_by_line())
    texts = split_text(docs)
    new_text = docs_to_str(texts)
    embeddings = get_embeddings()
    docsearch = import_pinecone_db(new_text,embeddings)
    print("The data is successfully imported from the links...:-)")

if __name__ == "__main__":
    # Code to be executed when the file is run as the main module
    load_by_chunks()
