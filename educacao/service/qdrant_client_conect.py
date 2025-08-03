import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "proj_edu")



class QdrantService:
    def __init__(self, host: str, api_key: str, collection_name: str, vector_size: int = 384):
        self.client = QdrantClient(url=host, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.ensure_collection()

    def ensure_collection(self):
        """Cria a coleção caso não exista"""
        exists = self.client.collection_exists(self.collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Coleção '{self.collection_name}' criada.")
        else:
            print(f"📦 Coleção '{self.collection_name}' já existe.")

 
    def list_collections(self):
        return self.client.get_collections()
    
    def config_retriever(self, docs, embeddings, collection_name):
        vectorstore = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=os.environ['QDRANT_HOST'],
            api_key=os.environ['QDRANT_API_KEY'],
            prefer_grpc=True,
            collection_name=collection_name,
            force_recreate=True,
        )
        return vectorstore.as_retriever()
    
    def get_retriever(self, qdrant_collection, embeddings):
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=os.environ['QDRANT_HOST'],
            api_key=os.environ['QDRANT_API_KEY'],
            collection_name=qdrant_collection,
        )
        return vectorstore.as_retriever(
            search_type = 'mmr',
            search_kwargs = {'k': 6, 'fetch_k': 10}
        )
    
    def get_context(self, retriever, topic):
        retrieved_docs = retriever.invoke(topic)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context
    
    
# Exemplo de uso
if __name__ == "__main__":
    ...
    # service = QdrantService(QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTION)
    # print(service.list_collections())