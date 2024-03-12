import os
import re
from bleach import clean

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

chroma_path = "/data/liuhanwen"

model_name = "/data/liuhanwen/models/bge-m3"
model_kwargs = {"device": "cuda:1"}
encode_kwargs = {
    "normalize_embeddings": True,
    "batch_size": 384,
}

hf_datapath = "/data/knowledge_graph/nell/nell_belief_sentences"

collection_name = "nell_belief_sentences_1"

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
    chunk_size=512,
    chunk_overlap=0,
    separator=" ",
)

def clean_brackets(text_list):
    # 定义一个空列表，用于存放处理过的文本
    cleaned_texts = []
    
    # 遍历输入的文本列表
    for text in text_list:
        # 使用正则表达式替换 [[ 和 ]]（及其周围的空格）为 ""
        cleaned_text = re.sub(r'\[\[\s*|\s*\]\]', '', text)
        # 将处理过的文本添加到列表中
        cleaned_texts.append(cleaned_text)
    
    return cleaned_texts

for i in tqdm(range(0, len(dataset), 384)):
    data = dataset[i: i + 384]
    # print(data["sentence"])
    # text = [d["sentence"] for d in data if "sentence" in d.keys()]
    datas = clean_brackets(data["sentence"])
    # print(datas)
    new_docs = text_splitter.create_documents(datas)
    # db.add_texts(
    #     texts=data["sentence"]
    # )
    try:
        db.add_documents(
            documents=new_docs
        )
    except:
        continue

print(db._collection.count())