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


def read_process(unparsed_file, queue, parallel):
    
    for file in unparsed_file:
        with open(file, 'r', encoding='utf-8', errors='ignore') as r:
            for line in r:
                try:
                    queue.put(line)
                except Exception as e:
                    continue

    for _ in range(parallel):
        queue.put(None)

    print(f"Read Done!", flush=True)


def handle(line, mid2name_dict):
    try:
        entity1, relation, entity2, _ = line.strip().split("\t")
        entity1 = entity1[entity1.rfind("/") + 1: -1]
        if entity1.startswith("m."):
            entity1 = mid2name_dict.get("/" + entity1.replace(".", "/"), "entity1 not found")
        if "\"" in entity2:
            entity2 = entity2[entity2.find("\"") + 1: entity2.rfind("\"")]
        else:
            entity2 = entity2[entity2.rfind("/") + 1: -1]
            if entity2.startswith("m."):
                entity2 = mid2name_dict.get("/" + entity2.replace(".", "/"), "entity2 not found")
        if "www.w3.org" not in relation:
            relation = relation[relation.rfind("/") + 1: -1]
        
        if entity1 and entity2 and relation:
            return f"({entity1}, {relation}, {entity2})"
        else:
            return None
    except:
        return None


def handle_process(index, queue, fout, error_count, correct_count, wlock, mid2name_dict):
    with tqdm(desc=multiprocessing.current_process().name, position=index) as pbar:
        count = 0
        while True:
            line = queue.get()
            if line is None:
                break

            obj = handle(line, mid2name_dict)
            if not obj:
                continue
            wlock.acquire()
            if "entity1 not found" in obj or "entity2 not found" in obj:
                error_count.value += 1
            else:
                correct_count.value += 1
                fout.write(obj + '\r\n')
                fout.flush()
            wlock.release()
            count += 1
            pbar.update(1)



if __name__ == "__main__":
    doc_dir_path = "/data/knowledge_graph/freebase"
    doc_name = {
        "kg": "freebase-rdf-latest",
        "mid2wiki": "fb2w.nt",
        "mid2name": "mid2name.txt",
        "test": "freebase-test"
    }
    
    w = open(os.path.join(doc_dir_path, "transed_data"), encoding='utf-8', mode='w')

    file_path = [os.path.join(doc_dir_path, doc_name["kg"])]
    
    mid2name_dict = {}
    with open(os.path.join(doc_dir_path, doc_name["mid2name"]), "r") as f:
        for line in f:
            try:
                mid, name = line.strip().split("\t")
                mid2name_dict[mid] = name
            except:
                continue

    queue = multiprocessing.Queue(maxsize=10 * 1000 * 1000)
    wlock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    error_count = manager.Value(int, 0)
    correct_count = manager.Value(int, 0)

    parallel = 36
    read_process = multiprocessing.Process(target=read_process, args=(file_path, queue, parallel))
    write_precesses = [
        multiprocessing.Process(target=handle_process, args=(i, queue, w, error_count, correct_count, wlock, mid2name_dict))
        for i in range(parallel)
    ]
    for process in write_precesses:
        process.start()
    read_process.start()

    for process in write_precesses:
        process.join()
    read_process.join()

    w.close()

    with open("trans_log.txt", encoding='utf-8', mode='w') as w:
        w.write(f"error count: {error_count.value}\n")
        w.write(f"correct count: {correct_count.value}\n")
        w.write(f"total count: {error_count.value + correct_count.value}\n")

    print(f"error count: {error_count.value}")
    print(f"correct count: {correct_count.value}")
    print(f"total count: {error_count.value + correct_count.value}")
