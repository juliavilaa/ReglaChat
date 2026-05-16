import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Importaciones de RAGAS
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

def evaluar_sistema():
    print("1. Cargando el sistema RAG (Base Vectorial y LLM)...")
    # Cargar modelos (Igual que en la app)
    embeddings_locales = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings_locales, 
        collection_name="reglamento_db"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.0)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # El mismo prompt de tu App
    PROMPT_TEMPLATE = """
    Eres un asistente institucional experto.
    Responde la pregunta basándote ÚNICAMENTE en este contexto: {context}
    Si no está la respuesta, di exactamente: "No encuentro esa información en el reglamento."
    Pregunta: {question}
    Respuesta:"""
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 2. DEFINIR LAS 10 PREGUNTAS DE PRUEBA (¡AQUÍ DEBES PONER LAS TUYAS!)
    muestras_evaluacion = [
        {
            "user_input": "¿Cuántas inasistencias causan la pérdida de la materia?",
            "reference": "Se reprueba al acumular un 20% de inasistencias injustificadas."
        },
        {
            "user_input": "¿Cuál es la nota mínima aprobatoria?",
            "reference": "La nota mínima aprobatoria es 3.0."
        },
        # ... ¡Agrega hasta llegar a 10 preguntas basadas en TU PDF! ...
    ]

    registros = []
    print("\n2. Generando respuestas del asistente para las preguntas de prueba...")
    for muestra in muestras_evaluacion:
        pregunta = muestra["user_input"]
        
        # Recuperar contexto
        docs = retriever.invoke(pregunta)
        contexto = "\n\n".join([d.page_content for d in docs])
        
        # Generar respuesta
        prompt_final = prompt_template.invoke({"context": contexto, "question": pregunta})
        respuesta_llm = llm.invoke(prompt_final).content
        
        # Guardar en el formato que exige RAGAS
        registros.append({
            "user_input": pregunta,
            "retrieved_contexts": [d.page_content for d in docs],
            "response": respuesta_llm,
            "reference": muestra["reference"]
        })
        print(f"   [OK] Pregunta procesada: {pregunta}")

    # 3. EVALUACIÓN CON RAGAS
    print("\n3. Iniciando el Juez RAGAS (Esto tomará unos minutos y consumirá tokens de Gemini)...")
    
    # RAGAS necesita envolver los modelos para usarlos como jueces
    llm_juez = LangchainLLMWrapper(llm)
    embeddings_juez = LangchainEmbeddingsWrapper(embeddings_locales)
    
    dataset = EvaluationDataset.from_list(registros)
    
    resultados = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm_juez,
        embeddings=embeddings_juez
    )

    # 4. EXPORTAR RESULTADOS
    print("\n==========================================")
    print("RESULTADOS DE EVALUACIÓN RAGAS")
    print("==========================================")
    df_resultados = resultados.to_pandas()
    
    # Mostrar en consola
    columnas_mostrar = ["user_input", "faithfulness", "answer_relevancy", "context_precision"]
    print(df_resultados[columnas_mostrar].to_string(index=False))
    
    # Guardar en un CSV para tu informe final
    df_resultados.to_csv("informe_evaluacion_rag.csv", index=False)
    print("\n¡Listo! Resultados guardados en 'informe_evaluacion_rag.csv'. Pega esto en tu documento.")

if __name__ == "__main__":
    evaluar_sistema()