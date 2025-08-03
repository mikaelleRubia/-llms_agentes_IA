from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

class UploadFiles:

    @staticmethod
    def parse_document(file_path):
        # carregamento do arquivo
        loader = DoclingLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_documents(input_path):
        # buscar todos os arquivos .pdf 
        input_path = Path(input_path)

        if input_path.is_dir():
            pdf_files = list(input_path.glob("*.pdf"))
        elif input_path.is_file():
            pdf_files = [input_path]
        else:
            raise ValueError("Caminho inválido. Forneça um diretório ou arquivo")

        documents = []
        for file in pdf_files:
            documents.extend(UploadFiles.parse_document(file))

        print(f"Documentos carregados: {len(documents)}")
        return documents
    

    @staticmethod
    def split_markdown(documents):
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3")
            ]
        )
        markdown_splits = [
            split
            for doc in documents
            for split in splitter.split_text(doc.page_content)
        ]

        print(f"Splits gerados via Markdown: {len(markdown_splits)}")
        return markdown_splits
    
    @staticmethod
    def split_chunks(markdown_splits, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(markdown_splits)
        print(f"Chunks gerados: {len(chunks)}")
        return chunks

    @staticmethod
    def build_chunks(input_path, chunk_size=1000, chunk_overlap=200):
        documents = UploadFiles.load_documents(input_path)
        markdown_splits = UploadFiles.split_markdown(documents)
        chunks = UploadFiles.split_chunks(markdown_splits, chunk_size, chunk_overlap)
        return chunks


# if __name__ == "__main__":
#     ...
    # print(UploadFiles.build_chunks("artigos/biologia.pdf"))
