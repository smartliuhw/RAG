import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TextSplitter

doc_dir_path = "/data/knowledge_graph/freebase"
doc_name = {
    "kg": "freebase-rdf-latest",
    "test": "freebase_test",
    "json_test": "alpaca_data.json",
}

chroma_path = "/data/liuhanwen/chroma"

model_name = "/data/liuhanwen/models/bge-m3"
model_kwargs = {"device": "cuda:1"}
encode_kwargs = {
    "normalize_embeddings": True,
    "batch_size": 2048,
}

collection_name = "langchain_test"

# hf = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True
)

# db = Chroma(collection_name=collection_name, persist_directory=chroma_path, embedding_function=hf)

raw_documents = TextLoader(os.path.join(doc_dir_path, doc_name["test"])).load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=128,
    chunk_overlap=0,
)

documents = text_splitter.split_documents(raw_documents)
print("len: ", len(documents))

db = Chroma.from_documents(
    documents=documents,
    embedding=hf,
    collection_name=collection_name,
    persist_directory=chroma_path,
)

print(db._collection.count())

embedding = hf.embed_query("Terry Tyler")
print(db.similarity_search_by_vector(embedding=embedding, k=2))