import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TextSplitter

from datasets import load_dataset
from tqdm import tqdm

doc_dir_path = "/data/knowledge_graph/freebase"
doc_name = {
    "kg": "freebase-rdf-latest",
    "test": "freebase_test",
    "json_test": "alpaca_data.json",
}

chroma_path = "/data/liuhanwen/chroma"

model_name = "/data/liuhanwen/models/bge-m3"
model_kwargs = {"device": "cuda"}
encode_kwargs = {
    "normalize_embeddings": True,
}

hf_datapath = "/data/knowledge_graph/nell/nell_belief_sentences"

collection_name = "langchain_saving_test"

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

dataset = load_dataset(hf_datapath)["train"]

db = Chroma(
    collection_name=collection_name,
    persist_directory=chroma_path,
    embedding_function=hf,
)

text_splitter = CharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=0,
)

for i in tqdm(range(0, len(dataset), 64)):
    data = dataset[i: i + 64]
    # print(data["sentence"])
    # text = [d["sentence"] for d in data if "sentence" in d.keys()]
    new_docs = text_splitter.create_documents(data["sentence"])
    # db.add_texts(
    #     texts=data["sentence"]
    # )
    db.add_documents(
        documents=new_docs
    )

print(db._collection.count())