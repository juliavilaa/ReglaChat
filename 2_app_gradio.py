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
        "k": 5  # AUMENTAR DRÁSTICAMENTE LA CANTIDAD DE CONTEXTO
    }
)
# --- DISEÑO DEL SYSTEM PROMPT (TFTCR + FEW-SHOT) ---
PROMPT_TEMPLATE = """
Eres un asistente institucional experto.

[TASK] Tu objetivo es responder consultas de estudiantes basándote ÚNICAMENTE en el reglamento institucional proporcionado.
[FORMAT] Responde de manera clara y directa. Usa viñetas o listas numeradas si la respuesta implica múltiples pasos o condiciones.
[TOPIC] Reglamento Estudiantil y normatividad académica.
[TONE] Institucional, profesional, respetuoso y útil.
[CONSTRAINTS/REQUIREMENTS]
1. NUNCA uses conocimiento externo.
2. Si la respuesta no está explícitamente en el [CONTEXT], debes responder EXACTAMENTE con esta frase: "No encuentro esa información en el reglamento."
3. No inventes excepciones ni asumas políticas que no estén escritas.

[FEW-SHOT EXAMPLES]
Pregunta: ¿Cuántas inasistencias puedo tener antes de perder la materia?
Contexto: "El estudiante reprobará la asignatura por fallas si acumula un porcentaje igual o superior al 20% de inasistencias injustificadas durante el semestre."
Respuesta: Se   gún el reglamento, reprobarás la asignatura si acumulas un 20% o más de inasistencias injustificadas durante el semestre académico.
---
Pregunta: ¿Dónde queda la cafetería principal?
Contexto: "Las instalaciones deportivas están ubicadas en el bloque B."
Respuesta: No encuentro esa información en el reglamento.

[CONTEXT]
{context}

[QUESTION]
{question}

Respuesta:
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

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