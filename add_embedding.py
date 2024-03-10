from __future__ import unicode_literals
from __future__ import print_function
# from base_udf import BaseUDF
import re
# from bs4 import BeautifulSoup, NavigableString, Comment, Tag
from multiprocessing import Process, Pipe
import multiprocessing
from tqdm import tqdm
import traceback
import json
import urllib

import urllib.request
import codecs
# import chardet
import os
import random

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TextSplitter

from datasets import load_dataset
from tqdm import tqdm
import torch

def read_process(dataset, queue, parallel):
    
    for i in range(len(dataset)):
        try:
            queue.put(dataset[i]["sentence"])
        except Exception as e:
            continue

    for _ in range(parallel):
        queue.put(None)

    print(f"Read Done!", flush=True)


def handle_process(index, queue, wlock):
    with tqdm(desc=multiprocessing.current_process().name, position=index) as pbar:
        count = 0
        while True:
            line = queue.get()
            if line is None:
                break

            db.add_texts(
                texts=[line]
            )
            wlock.release()
            count += 1
            pbar.update(1)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
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
        "batch_size": 16,
    }

    hf_datapath = "/data/knowledge_graph/nell/nell_belief_sentences"

    collection_name = "langchain_saving_test"

    hf = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=False
    )

    dataset = load_dataset(hf_datapath)["train"]

    db = Chroma(
        collection_name=collection_name,
        persist_directory=chroma_path,
        embedding_function=hf,
    )

    queue = multiprocessing.Queue(maxsize=10 * 1000 * 1000)
    wlock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    error_count = manager.Value(int, 0)
    correct_count = manager.Value(int, 0)

    parallel = 36
    read_process = multiprocessing.Process(target=read_process, args=(dataset, queue, parallel))
    run_processes = [
        multiprocessing.Process(target=handle_process, args=(i, queue, wlock))
        for i in range(parallel)
    ]
    for process in run_processes:
        process.start()
    read_process.start()

    for process in run_processes:
        process.join()
    read_process.join()
