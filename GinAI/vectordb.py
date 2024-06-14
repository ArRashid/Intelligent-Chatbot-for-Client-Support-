import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings



class PinconeDb():
    def __init__(self, embedding=None) -> None:
        if not embedding == None:
            if embedding == "OPENAI":
                self.embedding = self.get_openai_embeddings()
                print("Successfully created openai embedding for vectorDB")
            elif embedding == "HUGGINGFACE":
                self.embedding = self.get_huggingface_embeddings()
            else:
                self.embedding = embedding

    def get_docsearch(self, embedding=None, pinecone_api_key=None, pinecone_env=None, pinecone_index=None):
        if embedding == None:
            embedding = self.embedding
        pinecone.init(
            api_key=pinecone_api_key,  # find at app.pinecone.io
            environment=pinecone_env  # next to api key in console
        )
        if pinecone_index in pinecone.list_indexes():
            docsearch = Pinecone.from_existing_index(pinecone_index, embedding)
            print("Pinecone docsearch object for Index : {0} is created successfully !".format(
                pinecone_index))
            return docsearch

        else:
            print("Index : {0} is not foun in the picone database !".format(
                pinecone_index))

    def similasimilarity_search(self, query, embeddings=None, pinecone_api_key=None, pinecone_env=None, pinecone_index=None):
        if embeddings == None:
            embeddings = self.embedding

        self.pinecone_index = pinecone_index
        pinecone.init(
            api_key=pinecone_api_key,  # find at app.pinecone.io
            environment=pinecone_env  # next to api key in console
        )
        docsearch = self.get_docsearch()
        similar_docs = docsearch.similarity_search(query)
        return similar_docs

    def get_huggingface_embeddings(self, model_name, model_kwargs):
        #
        embeddings_out = HuggingFaceEmbeddings(model_name=model_name,
                                               model_kwargs=model_kwargs)
        return embeddings_out

    def get_openai_embeddings(self):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return embeddings


print("info : submadule: GinAI.vectordb imported successfully")
