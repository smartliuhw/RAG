from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
# from llama_index.storage.storage_context import StorageContext
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

embed_model = HuggingFaceEmbedding(model_path, embed_batch_size=6, device="cuda:1")

chroma_collection = db.get_or_create_collection("quick_test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context,
)

query_engine = index.as_query_engine()
retrival_engine = index.as_retriever()
query = "Today's weather is good"
# result = query_engine.query(query)
result = retrival_engine.retrieve(query)
# display(Markdown(f"<b>{result}</b>"))
print(result)
