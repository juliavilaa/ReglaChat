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
retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 15  # AUMENTAR DRÁSTICAMENTE LA CANTIDAD DE CONTEXTO
        }
    )

prompt_template = ChatPromptTemplate.from_template("""
Responde basado ÚNICAMENTE en el reglamento. Si no sabes, di "No encuentro esa información."
Contexto: {context}
Pregunta: {question}
""")

# 3. LA INTERFAZ MÁGICA
print("Iniciando interfaz web...")

with gr.Blocks() as interfaz:
    gr.Markdown("# 📚 Asistente del Reglamento Institucional")
    gr.Markdown("Hazme preguntas sobre el reglamento estudiantil.")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Escribe tu pregunta...", container=False)
    fuentes_checkbox = gr.Checkbox(label="Mostrar fuentes consultadas", value=False)
    with gr.Row():
        submit = gr.Button("Enviar", variant="primary")
        clear = gr.Button("Limpiar")
    fuentes_box = gr.Textbox(
        label="Fuentes consultadas",
        interactive=False,
        lines=8,
    )

    def responder_pregunta(mensaje, historial, mostrar_fuentes):
        if not mensaje.strip():
            return "", historial, ""

        docs = retriever.invoke(mensaje)
        contexto = "\n\n".join([d.page_content for d in docs])

        prompt = prompt_template.invoke({"context": contexto, "question": mensaje})
        respuesta = llm.invoke(prompt).content

        fuentes_texto = ""
        if mostrar_fuentes and docs:
            fuentes_texto = "\n".join(
                f"[{i+1}] {d.page_content[:200].strip().replace(chr(10), ' ')}..."
                for i, d in enumerate(docs)
            )

        historial = list(historial or [])
        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})
        return "", historial, fuentes_texto

    submit.click(responder_pregunta, inputs=[msg, chatbot, fuentes_checkbox], outputs=[msg, chatbot, fuentes_box])
    msg.submit(responder_pregunta, inputs=[msg, chatbot, fuentes_checkbox], outputs=[msg, chatbot, fuentes_box])
    clear.click(lambda: ([], ""), outputs=[chatbot, fuentes_box])

interfaz.launch()