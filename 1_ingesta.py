import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def crear_base_vectorial_local():
    print("1. Cargando el documento PDF...")
    loader = PyPDFLoader(r'pdfs\reglamento.pdf') 
    documentos = loader.load()

    print("2. Aplicando Chunking Mejorado...")
    # Aumentamos el tamaño y usamos los "Artículos" como separadores principales
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,    # Aumentado para capturar artículos completos
        chunk_overlap=200,  # Aumentado para mantener la coherencia
        separators=["\nARTÍCULO", "\nPARÁGRAFO", "\n\n", "\n", ".", " "]
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"   -> Se generaron {len(fragmentos)} chunks.")

    print("3. Generando Embeddings Multilingües...")
    # Este modelo es excelente para textos en español
    embeddings_locales = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("4. Guardando en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings_locales,
        persist_directory="./chroma_db",
        collection_name="reglamento_db"
    )
    print("¡Base vectorial creada con éxito!")

if __name__ == "__main__":
    crear_base_vectorial_local()