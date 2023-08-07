from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.indexes import VectorstoreIndexCreator
from detectron2.config import get_cfg
import os

cfg = get_cfg()    
cfg.MODEL.DEVICE = 'gpu' #GPU is recommended

text_folder = 'gi_datasets'
loaders = [UnstructuredFileLoader(os.path.join(text_folder, fn)) for fn in os.listdir(text_folder)]

hf_repo = "IIC/roberta-base-spanish-sqac"

index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings(), text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

llm = HuggingFaceHub(repo_id=hf_repo, 
                     model_kwargs={"truncation": "only_first"},
                     huggingfacehub_api_token="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", 
                     task="text2text-generation")

chain = RetrievalQA.from_chain_type(llm=llm,  chain_type="stuff",  retriever=index.vectorstore.as_retriever(), input_key="question")

resp = chain.run('¿Quién inventó el submarino?')

print(resp)