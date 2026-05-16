import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings # Embeddings locales
from langchain_google_genai import ChatGoogleGenerativeAI # LLM en la nube
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="Asistente RAG - Reglamento", layout="centered")
st.title("📚 Asistente Institucional (Híbrido)")

@st.cache_resource
def inicializar_sistema():
    # 1. Recuperador Local (Mismo modelo que usamos en la ingesta)
    embeddings_locales = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings_locales,
        collection_name="reglamento_db"
    )
    
    # 2. Generador en la Nube (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0 # CRÍTICO: 0.0 para que sea analítico y no invente nada
    )
    return vector_store, llm

vector_store, llm = inicializar_sistema()

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
Respuesta: Según el reglamento, reprobarás la asignatura si acumulas un 20% o más de inasistencias injustificadas durante el semestre académico.
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

# --- LÓGICA DE INTERACCIÓN ---
pregunta = st.text_input("Haz una pregunta sobre el reglamento:")

if st.button("Consultar") and pregunta:
    with st.spinner("Analizando similitud de vectores y generando respuesta..."):
        # 1. Recuperar los 4 chunks más relevantes (Búsqueda local)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        documentos_recuperados = retriever.invoke(pregunta)
        
        # 2. Unir los chunks en un solo texto
        contexto = "\n\n---\n\n".join([doc.page_content for doc in documentos_recuperados])
        
        # 3. Formatear el Prompt y enviarlo a Gemini
        prompt_final = prompt_template.invoke({"context": contexto, "question": pregunta})
        respuesta = llm.invoke(prompt_final)
        
        # 4. Mostrar Resultados en la GUI
        st.success(respuesta.content)
        
        # Transparencia: Mostrar de dónde sacó la información
        with st.expander("🔍 Ver fragmentos (chunks) recuperados del documento"):
            for i, doc in enumerate(documentos_recuperados, 1):
                st.info(f"**Chunk {i}** (Pág. {doc.metadata.get('page', 'Desconocida')}):\n{doc.page_content}")