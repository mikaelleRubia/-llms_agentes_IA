from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client_conect import QdrantService
import os

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "proj_edu_v2")

class Embedding():
    def __init__(self, embedding_model:str='BAAI/bge-m3'):
        """
        Inicializa a classe com o modelo Hugging Face desejado.
        """
        self.model_name = embedding_model
        self._embedding = None

    def get_model(self):
        """
        Retorna uma instância de HuggingFaceEmbeddings do LangChain.
        """
        if self._embedding is None:
            self._embedding = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embedding

if __name__ == "__main__":

    serviceqdrant = QdrantService(QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTION)
    service_embed = Embedding('BAAI/bge-m3')

    # text = """
    # A biblioteca possui atualmente 42998 livros.
    # """
    # # transformando o text no tipo Document
    # docs = [Document(page_content=text)]

    # chunks = UploadFiles.build_chunks("artigos/biologia.pdf", chunk_size=800, chunk_overlap=100)
    
    collections = serviceqdrant.client.get_collections().collections
    collection_name = collections[0].name if collections else "default_collection"

    # retriever = serviceqdrant.config_retriever(chunks, service_embed.get_model(), collection_name)

    retiever = serviceqdrant.get_retriever(collection_name, service_embed.get_model())

    context = serviceqdrant.get_context(retiever,  'biblioteca')
    print(context)