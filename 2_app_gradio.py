import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. ESTO CARGA UNA SOLA VEZ (Adiós crasheos silenciosos)
print("Cargando IA...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="reglamento_db")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.0)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

prompt_template = ChatPromptTemplate.from_template("""
Responde basado ÚNICAMENTE en el reglamento. Si no sabes, di "No encuentro esa información."
Contexto: {context}
Pregunta: {question}
""")

# 2. LA FUNCIÓN DEL CHAT
def responder_pregunta(mensaje, historial):
    # Buscar en Chroma
    docs = retriever.invoke(mensaje)
    contexto = "\n\n".join([d.page_content for d in docs])
    
    # Preguntar a Gemini
    prompt = prompt_template.invoke({"context": contexto, "question": mensaje})
    respuesta = llm.invoke(prompt).content
    return respuesta

# 3. LA INTERFAZ MÁGICA
print("Iniciando interfaz web...")
interfaz = gr.ChatInterface(
    fn=responder_pregunta,
    title="📚 Asistente del Reglamento Institucional",
    description="Hazme preguntas sobre el reglamento estudiantil.",
    # type="messages" # <- Agrega esto en lugar del theme
)

interfaz.launch()