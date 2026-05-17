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
    
    # 1. USAR EL NUEVO MODELO MULTILINGÜE
    embeddings_locales = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings_locales, 
        collection_name="reglamento_db"
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.0)
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 8,          # Devuelve los 8 mejores fragmentos al LLM
            "fetch_k": 30,   # Primero busca los 30 más similares en la base
            "lambda_mult": 0.7 # Equilibrio entre similitud (1.0) y diversidad (0.0)
        }
    )

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
        "reference": "Se reprueba la asignatura cuando el estudiante acumula un porcentaje de inasistencia igual o superior al 20% del total de horas de la asignatura en el período académico (o al 15% en el caso de prácticas profesionales). La asignatura se califica con cero (0) con la anotación 'reprobada por inasistencia'."
    },
    {
        "user_input": "¿Cuál es la nota mínima aprobatoria ordinaria?",
        "reference": "La nota mínima aprobatoria ordinaria es de treinta (30) puntos en una escala de cero (0) a cincuenta (50), excepto en el caso de repetición de asignaturas, de la práctica profesional, el trabajo de grado y de las actividades académicas que determine el programa."
    },
    {
        "user_input": "¿Qué pasa si un estudiante pierde una asignatura por tercera vez?",
        "reference": "Cuando el estudiante reprueba por tercera vez una asignatura, pierde el cupo en el programa académico que esté cursando. Podrá solicitar su reingreso al Consejo de la Facultad/Escuela, órgano que estudiará y decidirá sobre la solicitud."
    },
    {
        "user_input": "¿Cuántos créditos equivale un crédito académico en horas?",
        "reference": "Un crédito académico equivale a 48 horas de trabajo académico del estudiante, que comprende las horas con acompañamiento directo del docente (sincrónicas presenciales o remotas mediadas por TIC) y las demás horas que el estudiante debe emplear en actividades independientes de estudio, prácticas u otras necesarias para alcanzar los objetivos de aprendizaje, sin incluir las destinadas a la presentación de las pruebas finales de evaluación."
    },
    {
        "user_input": "¿Cuáles son los requisitos para obtener una Mención de Honor?",
        "reference": "Para obtener una Mención de Honor el estudiante debe: 1) Haber cursado en el semestre correspondiente la carga académica completa. 2) Haber obtenido un promedio de calificación mínimo de cuarenta y tres (43) en una escala de cero (0) a cincuenta (50) sin perder ninguna asignatura en el semestre. 3) No haber tenido sanciones disciplinarias durante su permanencia en la Facultad/Escuela."
    },
    {
        "user_input": "¿Cuáles son los documentos requeridos para inscribirse a un programa de pregrado?",
        "reference": "Los aspirantes deben: 1) Entregar el formulario de inscripción debidamente diligenciado. 2) Entregar copia del diploma de bachiller debidamente reconocido por el Estado. 3) Entregar el certificado del examen de Estado. 4) Entregar copia del documento de identidad vigente. 5) Adjuntar el recibo de cancelación de derechos de inscripción por la suma fijada por el Consejo Superior."
    },
    {
        "user_input": "¿Qué porcentaje máximo puede tener un examen final sobre la nota total?",
        "reference": "Los exámenes finales no podrán tener un valor mayor al 25% del total de la nota. Igualmente, en ningún caso el porcentaje asignado a cada prueba y actividad académica podrá exceder el 25% del total de la nota."
    },
    {
        "user_input": "¿Cuánto tiempo tiene un estudiante antiguo para solicitar devolución del dinero de matrícula si se retira?",
        "reference": "El estudiante antiguo dispone de quince (15) días calendario después de iniciadas las labores académicas correspondientes como fecha límite para notificar su retiro a la Dirección del Programa y solicitar la devolución del valor cancelado. Los estudiantes antiguos que se retiren dentro de ese plazo tienen derecho a la devolución del 80% del valor total de la matrícula. Los estudiantes que ingresan al primer período no tienen derecho a reembolso."
    },
    {
        "user_input": "¿Cuáles son los tipos de práctica profesional reconocidos por la institución?",
        "reference": "La institución reconoce tres tipos de práctica profesional: 1) Prácticas de preparación profesional: vinculan al estudiante a una entidad pública o privada donde desempeña funciones relacionadas con el ejercicio profesional. 2) Prácticas investigativas: vinculan al estudiante a una experiencia de formación de tipo investigativo en instituciones de educación superior o centros de investigación. 3) Prácticas de autogestión: fomentan el espíritu emprendedor mediante la realización de la práctica en empresa propia, familiar o a través de un plan de negocios."
    },
    {
        "user_input": "¿Cuáles son los requisitos para graduarse en la Fundación Universitaria Konrad Lorenz?",
        "reference": "Para obtener el grado, el estudiante debe: 1) Haber cursado y aprobado todas las asignaturas correspondientes a su plan de estudios. 2) Presentar y aprobar los requisitos exigidos por cada programa en su registro calificado. 3) Haber presentado las pruebas formativas y el Examen Saber Pro. 4) Estar a paz y salvo con la Institución por todo concepto. 5) Realizar el pago de los derechos de grado."
    },
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
        respuesta_cruda = llm.invoke(prompt_final).content
        
        # Si Gemini nos devuelve una lista de bloques, extraemos solo el texto
        if isinstance(respuesta_cruda, list):
            respuesta_llm = "".join([bloque.get("text", "") for bloque in respuesta_cruda if isinstance(bloque, dict)])
        else:
            # Si ya es un texto normal, lo aseguramos como string
            respuesta_llm = str(respuesta_cruda)
        
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