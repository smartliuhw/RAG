import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
# from llama_index.storage.storage_context import StorageContext
from llama_index.core import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

doc_dir_path = "/data/knowledge_graph"
doc_name = {
    "kg": "freebase-rdf-latest",
    "test": "alpaca_data.json",
}

model_path = "/data/liuhanwen/models/bge-m3"

chroma_path = "/data/liuhanwen/chroma"

db = chromadb.PersistentClient(chroma_path)
# db = chromadb.Client()
chroma_collection = db.get_or_create_collection("quick_test")

# print(db.list_collections())
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = db.get_or_create_collection("quick_test")

embed_model = HuggingFaceEmbedding(model_path, embed_batch_size=6, device="cuda:1", max_length=512)
print("model: ", embed_model)

documents = SimpleDirectoryReader(input_files=[os.path.join(doc_dir_path, doc_name["test"])]).load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)
