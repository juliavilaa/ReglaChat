import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # IMPORTANTE: Modelo Local
from langchain_chroma import Chroma

def crear_base_vectorial_local():
    print("1. Cargando el documento PDF...")
    loader = PyPDFLoader(r'pdfs\reglamento.pdf') # Asegúrate de que el PDF esté aquí
    documentos = loader.load()

    print("2. Aplicando Chunking (Dividiendo en fragmentos)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # Tamaño ideal para capturar una regla completa
        chunk_overlap=50,  # Solapamiento para no cortar ideas por la mitad
        separators=["\n\n", "\n", ".", " "]
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"   -> Se generaron {len(fragmentos)} chunks.")

    print("3. Generando Embeddings Locales (all-MiniLM-L6-v2)...")
    # Este modelo se descargará la primera vez que lo corras (pesa muy poco, ~90MB)
    embeddings_locales = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("4. Guardando en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings_locales,
        persist_directory="./chroma_db",
        collection_name="reglamento_db"
    )
    print("¡Base vectorial híbrida creada con éxito!")

if __name__ == "__main__":
    crear_base_vectorial_local()