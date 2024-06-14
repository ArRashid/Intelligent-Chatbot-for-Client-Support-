from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import WebBaseLoader,PyPDFLoader,UnstructuredPDFLoader,CSVLoader
from . import * 

class Loader():
	#create obj
	#laod Documents
	#split text
	#load into db
	
	def __init__(self,chunk_size=DEFAULT_CHUNK_SIZE,chunk_overlap=DEFAULT_CHUNK_OVERLAP,embedding=EMBEDDING):
		print("Created a Loader object with paramiter chunk_size={0},chunk_overlap={1},embedding={2}".format(chunk_size,chunk_overlap,embedding))
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.documents = []
		#checking the embbdibg as getting embedding
		if not embedding == None:
			if embedding == "OPENAI":
				self.embedding = self.get_openai_embeddings()
			elif embedding == "HUGGINGFACE":
				self.embedding = self.get_huggingface_embeddings()
		

	# convert the docs to splited texts
	def create_splited_text(self,documents=None,chunk_size=None,chunk_overlap=None):
		print("Loader.create_splited_text is called with prams : documents={0},chunk_size={1},chunk_overlap={2} ".format(documents,chunk_size,chunk_overlap))
		if chunk_size is None:
			chunk_size = self.chunk_size
		if chunk_overlap is None:
			chunk_overlap = self.chunk_overlap
		if documents is None:
			documents = self.documents
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
		self.splited_texts = text_splitter.split_documents(documents)
		print(f"Split into {len(self.splited_texts)} chunks of text (max. {chunk_size} tokens each)")
		return self.splited_texts
	
	#Import into vctor database 
	def import_into_db(self):
		if DATABASE ==  "PINECONE":
			self.load_data_into_pinconedb(self.splited_texts)



		
	def load_data_into_chromadb(self):
		pass
	def load_data_into_pinconedb(self,list_of_texts,embeddings=None,pinecone_api_key=PINECONE_API_KEY,pinecone_env=PINECONE_ENV,pinecone_index=PINECONE_INDEX_NAME)	:
		print("Loader.load_data_into_pinconedb is called with prams len(list_of_texts)={0},embeddings={1},pinecone_api_key={2},pinecone_env={3},pinecone_index={4} ".format(len(list_of_texts),embeddings,pinecone_api_key,pinecone_env,pinecone_index))
		if embeddings  == None:
			embeddings = self.embedding
		self.pinecone_api_key = pinecone_api_key
		self.pinecone_env = pinecone_env
		self.pinecone_index = pinecone_index
		pinecone.init(
        api_key=self.pinecone_api_key,  # find at app.pinecone.io
        environment=self.pinecone_env  # next to api key in console
    	)

		#check the index is present or not ( if not create that)
		if not pinecone_index in pinecone.list_indexes():
			pinecone.create_index(self.pinecone_index, dimension=128)
		
		self.dbindex = Pinecone.from_texts(list_of_texts, embeddings, index_name=pinecone_index)
		return self.dbindex
	
	
		#check the index is present or not ( if not create that)
		if  pinecone_index in pinecone.list_indexes():
			dbobj = Pinecone.from_existing_index(pinecone_index, embeddings)

		return dbobj



	def get_huggingface_embeddings(self,model_name=HUGGINGFACE_MODEL_NAME,model_kwargs=HUGGINGFACE_MODEL_KWARGS):
		from langchain.embeddings import HuggingFaceEmbeddings
		embeddings_out = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs)
		return embeddings_out
	

	def get_openai_embeddings(self,openai_api_key=OPENAI_API_KEY):
		print("Loder.get_openai_embeddings is called with pram openai_api_key={0} ".format(openai_api_key))
		embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
		return embeddings

	def convert_splited_texts_into_list(self,splited_texts=None):
		if splited_texts is None:
			splited_texts = self.splited_texts
		list_of_text = []
		for text in splited_texts :
			text_new = str(text)
			list_of_text.append(text_new)
		self.splited_texts = list_of_text
		return list_of_text
	
	def load_documents_from_links(self,links=None):
		print("starting loading total {0} web page  data ....".format(len(links)))
		loader = WebBaseLoader(links)
		loader.requests_per_second = 1
		new_documents = loader.load()
		print(f"The document has loded count of documents is {len(new_documents)} pcs .")
		return self.documents.extend(new_documents)
	def load_documents_from_text(self,text):
		pass

	def load_documents_from_texts(self,texts):
		pass

	def load_documents_from_pdf_perpage(self,paths=None):
		if len(paths) >= 1 :
			for path in paths :
				loader = PyPDFLoader(path)			
				new_documents = loader.load_and_split()
				self.documents.extend(new_documents)
		else: 
			print("Invalid oparation : The PDF file is less than One")

	def load_documents_from_pdf_unstructured(self,paths=None):
		if len(paths) >= 1 :
			for path in paths :
				loader = UnstructuredPDFLoader()			
				new_documents = loader.load(path,mode="elements")
				self.documents.extend(new_documents)
		else: 
			print("Invalid oparation : The PDF file is less than One")
	def load_documents_from_csv(self,paths=None):
		if len(paths) >= 1 :
			for path in paths :
				loader = CSVLoader(file_path=path)			
				new_documents = loader.load()
				self.documents.extend(new_documents)
		else: 
			print("Invalid oparation : The CSV  is less than One")

	def load_documents_from_csv_unstructured(self,paths=None):
		if len(paths) >= 1 :
			for path in paths :
				loader = UnstructuredCSVLoader(file_path=path,mode="elements")			
				new_documents = loader.load()
				self.documents.extend(new_documents)
		else: 
			print("Invalid oparation : The CSV file  is less than One")
		




	# links = ["https://www.ginesys.in/products/retail-erp","https://www.ginesys.in/products/business-intelligence","https://www.ginesys.in/products/retail-erp/warehouse-management","https://www.ginesys.in/products/integrations","https://www.ginesys.in/products/e-commerce-oms","https://www.ginesys.in/products/retail-erp/warehouse-management","https://www.ginesys.in/products/ecommerce-development-marketing","https://www.ginesys.in/products/cloud-pos","https://www.ginesys.in/products/desktop-pos","https://www.ginesys.in/products/gst-solutions","https://www.ginesys.in/products/gst-reconciliation","https://www.ginesys.in/products/e-documents"]
# link = ["https://www.ginesys.in/about-us"]
# objec = Loader()
# print("the loder object created successfully")
# objec.load_documents_from_links(links=link)
# print("The documents are created succssfully ")
# objec.create_splited_text()
# print("The text is now splited")
# objec.convert_splited_texts_into_list()
# print("the text is convarted into list of text")
# objec.import_into_db()
# print("All done imported into the dataae ###############")





#need to implement this into the this madule 
#https://python.langchain.com/en/latest/reference/modules/document_loaders.html
# from langchain.document_loaders import ConfluenceLoader

# loader = ConfluenceLoader(
#     url="https://yoursite.atlassian.com/wiki",
#     username="me",
#     api_key="12345"
# )

# documents = loader.load(space_key="SPACE",limit=50)


