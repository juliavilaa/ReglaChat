# ChatReglamento - Asistente de Consulta del Reglamento Institucional

Sistema RAG (Retrieval-Augmented Generation) que permite realizar consultas en lenguaje natural sobre el Reglamento Institucional de la Fundación Universitaria Konrad Lorenz, utilizando un documento PDF como fuente de conocimiento y generando respuestas fundamentadas exclusivamente en su contenido.

---

## Tabla de Contenidos

1. [Descripción de la Solución](#descripción-de-la-solución)
2. [Arquitectura](#arquitectura)
3. [Flujo del Proceso](#flujo-del-proceso)
4. [Ingesta de Documentos](#ingesta-de-documentos)
5. [Vectorización](#vectorización)
6. [Construcción del Prompt Aumentado](#construcción-del-prompt-aumentado)
7. [Evaluación e Informe de Resultados](#evaluación-e-informe-de-resultados)
8. [Requisitos e Instalación](#requisitos-e-instalación)
9. [Uso](#uso)
10. [Estructura del Proyecto](#estructura-del-proyecto)

---

## Descripción de la Solución

ChatReglamento es un chatbot inteligente basado en la arquitectura RAG que permite a estudiantes, docentes y personal administrativo consultar de forma conversacional el contenido del Reglamento Institucional. El sistema:

- **Ingiere** el documento PDF del reglamento y lo procesa en fragmentos semánticos
- **Vectoriza** cada fragmento usando embeddings multilingüñes optimizados para español
- **Recupera** los fragmentos más relevantes ante cada consulta del usuario
- **Genera** respuestas precisas utilizando Google Gemini, restringidas únicamente al contenido del reglamento
- **Evalúa** la calidad del sistema mediante el framework RAGAS con métricas de fidelidad, relevancia y precisión del contexto

---

## Arquitectura

```
┌────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA RAG                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐    ┌────────────┐    ┌────────────────┐   │
│  │  PDF      │───>│ PyPDF     │───>│ Text Splitter   │   │
│  │ reglamento  │    │ Loader    │    │ (chunking)      │   │
│  └────────────┘    └────────────┘    └──────────┬───────┘   │
│                                              │            │
│                                              v            │
│                                    ┌────────────────┐   │
│                                    │ Embeddings    │   │
│                                    │ paraphrase-   │   │
│                                    │ multilingual  │   │
│                                    │ MiniLM-L12    │   │
│                                    └───────────┬───────┘   │
│                                             │         │
│                                             v         │
│                                    ┌────────────────┐   │
│                                    │  ChromaDB     │   │
│                                    │  (vector DB   │   │
│                                    │   persistida) │   │
│                                    └───────────┬───────┘   │
│                                             │         │
│  ┌────────────┐    ┌────────────┐    │         │   │
│  │  Usuario  │───>│ Gradio UI │───>│ Retriever │   │
│  │ Pregunta  │    └────────────┘    └────────┬───────┘   │
│  └────────────┘                        │         │   │
│        │                                v         │   │
│        │                        ┌────────────────┐   │   │
│        │                        │ Prompt +    │   │   │
│        │                        │ Contexto    │───┼───│   │
│        │                        │ Aumentado   │   │   │
│        │                        └────────────────┘   │   │
│        │                                │         │   │
│        │                                v         │   │
│        │                        ┌────────────────┐   │   │
│        └───────────────────────────>│ Gemini LLM  │───┼───│   │
│                                 │ (respuesta) │   │   │
│                                 └────────────────┘   │   │
│                                             │         │   │
│  ┌────────────┐    ┌────────────┐    │         │   │
│  │ RAGAS     │<───│ Evaluación│<────────────────┘   │   │
│  │ (métricas) │    │          │                │   │
│  └────────────┘    └────────────┘                │   │
│                                               │   │
│  ┌────────────┐                                │   │
│  │ Informe   │<──────────────────────────────────────┘   │
│  │ CSV       │                                    │
│  └────────────┘                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Componentes Principales

| Componente | Tecnología | Función |
|---|---|---|
| **Parser PDF** | PyPDFLoader | Extracción de texto del documento |
| **Text Splitter** | RecursiveCharacterTextSplitter | Segmentación semántica por artículos |
| **Embeddings** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Vectorización multilingüe |
| **Vector DB** | ChromaDB | Almacenamiento y búsqueda por similitud |
| **LLM** | Google Gemini (gemini-3.1-flash-lite-preview) | Generación de respuestas |
| **UI** | Gradio (gr.Blocks) | Interfaz web conversacional con chatbot y controles |
| **Fuentes** | gr.Checkbox + gr.Textbox | Toggle para mostrar los fragmentos consultados del reglamento |
| **Evaluación** | RAGAS | Métricas de calidad del sistema RAG |
| **Orquestación** | LangChain | Pipeline RAG unificado |

---

## Flujo del Proceso

El sistema opera en tres etapas secuenciales:

### Etapa 1: Ingesta y Vectorización (ejecución única)
1. Lectura del PDF `pdfs/reglamento.pdf`
2. Segmentación en chunks con preservación de estructura legal
3. Generación de embeddings multilingües
4. Almacenamiento en ChromaDB persistente

### Etapa 2: Consulta Conversacional (servicio continuo)
1. El usuario formula una pregunta en la interfaz Gradio
2. La pregunta se vectoriza y se busca en ChromaDB (top-5 chunks)
3. Se construye el prompt aumentado con el contexto recuperado
4. Gemini genera la respuesta restringida al reglamento
5. La respuesta se muestra al usuario

### Etapa 3: Evaluación (bajo demanda)
1. Se ejecutan 10 preguntas de prueba con respuestas de referencia
2. RAGAS evalúa faithfulness, answer_relevancy y context_precision
3. Se genera el informe `informe_evaluacion_rag.csv`

---

## Ingesta de Documentos

El proceso de ingesta (`1_ingesta.py`) transforma el documento PDF en una base de conocimiento vectorizada:

### Configuración de Chunking

| Parámetro | Valor | Justificación |
|---|---|---|
| **Chunk Size** | 1200 caracteres | Balance entre contexto y precisión |
| **Chunk Overlap** | 200 caracteres | Preserva contexto entre fragmentos |
| **Separadores** | `["ARTICULO", "PARAGRAFO", "", "", ".", " "]` | Respeta la estructura legal del documento |

### Estrategia de Segmentación

El `RecursiveCharacterTextSplitter` aplica los separadores en orden de prioridad:
1. Intenta dividir en límites de `
ARTICULO` (preserva artículos completos)
2. Si no cabe, usa `
PARAGRAFO` (preserva parágrafos)
3. Falls back a separadores genéricos (``, ``, `.`, ` `)

Esto asegura que cada chunk mantenga coherencia semántica dentro del contexto regulatorio.

---

## Vectorización

### Modelo de Embeddings

- **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Idiomas:** Optimizado para 50+ idiomas incluyendo español
- **Dimensión:** 384 dimensiones por vector
- **Backend:** PyTorch con inferencia local

### Proceso

```
Texto del chunk ──> Tokenización ──> Modelo Transformer ──> Vector 384d ──> ChromaDB
```

Cada fragmento de texto se convierte en un vector denso que captura su significado semántico, permitiendo búsquedas por similitud coseno durante la recuperación.

### Base de Datos Vectorial

- **Motor:** ChromaDB con persistencia en disco
- **Ubicación:** `./chroma_db/`
- **Colección:** `reglamento_db`
- **Índice:** HNSW (Hierarchical Navigable Small World) para búsqueda eficiente
- **Métrica:** Distancia coseno (L2)

---

## Construcción del Prompt Aumentado

El prompt aumentado combina la pregunta del usuario con el contexto recuperado de la base vectorial:

### Template del Prompt (TFTCR + Few-Shot)

```
Eres un asistente institucional experto.

[TASK] Tu objetivo es responder consultas de estudiantes basandote UNICAMENTE en el reglamento institucional proporcionado.
[FORMAT] Responde de manera clara y directa. Usa viÃ±etas o listas numeradas si la respuesta implica multiples pasos o condiciones.
[TOPIC] Reglamento Estudiantil y normatividad academica.
[TONE] Institucional, profesional, respetuoso y util.
[CONSTRAINTS/REQUIREMENTS]
1. NUNCA uses conocimiento externo.
2. Si la respuesta no esta explicitamente en el [CONTEXT], debes responder EXACTAMENTE con esta frase: "No encuentro esa informacion en el reglamento."
3. No inventes excepciones ni asumas politicas que no esten escritas.

[FEW-SHOT EXAMPLES]
Pregunta: Cuantas inasistencias puedo tener antes de perder la materia?
Contexto: "El estudiante reprobara la asignatura por fallas si acumula un porcentaje igual o superior al 20% de inasistencias injustificadas durante el semestre."
Respuesta: Segun el reglamento, reprobaras la asignatura si acumulas un 20% o mas de inasistencias injustificadas durante el semestre academico.
---
Pregunta: Donde queda la cafeteria principal?
Contexto: "Las instalaciones deportivas estan ubicadas en el bloque B."
Respuesta: No encuentro esa informacion en el reglamento.

[CONTEXT]
{context}

[QUESTION]
{question}

Respuesta:
```

### Componentes del Prompt

| Elemento | Descripción |
|---|---|
| **Instrucción de restricción** | Obliga al LLM a usar solo el contexto proporcionado |
| **Fallback** | Instruye responder "No encuentro esa información" si no hay datos suficientes |
| **{context}** | Los 5 chunks más relevantes recuperados de ChromaDB |
| **{question}** | La pregunta formulada por el usuario |

### Parámetros del LLM

| Parámetro | Valor | Propósito |
|---|---|---|
| **Modelo** | `gemini-3.1-flash-lite-preview` | Balance velocidad/calidad |
| **Temperature** | `0.0` | Respuestas deterministas y consistentes |
| **k (retriever)** | `5` | Top-5 chunks más relevantes (equilibrio cobertura/precisión) |
| **Tipo de búsqueda** | `similarity` | Similitud coseno en el espacio vectorial |
| **max_output_tokens** | `2048` | Respuestas completas sin truncamiento |

### Flujo de Construcción

```
1. Usuario envía pregunta → "?¿Cuál es la nota mínima para aprobar?"
                                    ↓
2. Pregunta se vectoriza → Embedding de la consulta
                                    ↓
3. Búsqueda en ChromaDB → Top-5 chunks más similares
                                    ↓
4. Concatenación de contexto → Todos los chunks unidos en un solo string
                                    ↓
5. Inyección en el template → Prompt completo con contexto + pregunta
                                    ↓
6. Envío a Gemini → Generación de respuesta fundamentada
                                    ↓
7. Respuesta al usuario → "La nota mínima para aprobar es 30 sobre 50..."
```

---

## Evaluación e Informe de Resultados

El sistema incluye un pipeline de evaluación basado en el framework **RAGAS** que mide la calidad del sistema RAG con tres métricas fundamentales:

### Métricas Evaluadas

| Métrica | Descripción | Rango |
|---|---|---|
| **Faithfulness** | ¿La respuesta está fundamentada en el contexto recuperado? | 0.0 - 1.0 |
| **Answer Relevancy** | ¿La respuesta es relevante para la pregunta? | 0.0 - 1.0 |
| **Context Precision** | ¿Los contextos recuperados son relevantes para la pregunta? | 0.0 - 1.0 |

### Dataset de Evaluación

El archivo `3_evaluacion_ragas.py` contiene 10 preguntas de prueba con respuestas de referencia que cubren temas clave del reglamento:

| # | Tema | Pregunta |
|---|---|---|
| 1 | Inasistencias | ¿Cuál es el porcentaje de inasistencias que puede causar la pérdida del curso? |
| 2 | Nota mínima | ¿Cuál es la nota mínima para aprobar un curso? |
| 3 | Reprobación | ¿Qué pasa si un estudiante reprueba tres veces un curso? |
| 4 | Horas académicas | ¿Cuántas horas académicas tiene un crédito? |
| 5 | Mención honorífica | ¿Cuál es el promedio para mención honorífica? |
| 6 | Matrícula pregrado | ¿Qué documentos se requieren para la matrícula de pregrado? |
| 7 | Peso examen final | ¿Cuál es el porcentaje máximo que puede valer un examen final? |
| 8 | Retiro y reembolso | ¿En cuántos días se puede pedir reembolso al retirarse? |
| 9 | Prácticas profesionales | ¿Cuántos tipos de práctica profesional existen? |
| 10 | Graduación | ¿Cuáles son los requisitos para graduarse? |

### LLM como Juez

RAGAS utiliza Gemini como juez de evaluación mediante `LangchainLLMWrapper`, comparando las respuestas generadas contra las respuestas de referencia y el contexto recuperado.

### Generación del Informe

Los resultados se exportan a `informe_evaluacion_rag.csv` con las siguientes columnas:
- `question`: Pregunta de prueba
- `answer`: Respuesta generada por el sistema
- `contexts`: Contextos recuperados
- `ground_truth`: Respuesta de referencia
- `faithfulness`: Puntuación de fidelidad
- `answer_relevancy`: Puntuación de relevancia
- `context_precision`: Puntuación de precisión del contexto

### Ejecución de la Evaluación

```powershell
.\env\Scripts\python.exe 3_evaluacion_ragas.py
```

---

### Informe de Resultados

A continuación se presentan los resultados de la evaluación RAGAS sobre un dataset de 10 preguntas de prueba representativas del Reglamento Institucional:

#### Tabla de Resultados

| # | Pregunta | Faithfulness | Answer Relevancy | Context Precision |
|---|---|:---:|:---:|:---:|
| 1 | ¿Cuántas inasistencias causan la pérdida de la materia? | **1.00** | 0.54 | 0.00 |
| 2 | ¿Cuál es la nota mínima aprobatoria ordinaria? | **1.00** | 0.86 | **0.75** |
| 3 | ¿Qué pasa si un estudiante pierde una asignatura por tercera vez? | **1.00** | 0.00 | 0.00 |
| 4 | ¿Cuántos créditos equivale un crédito académico en horas? | **1.00** | 0.95 | **1.00** |
| 5 | ¿Cuáles son los requisitos para obtener una Mención de Honor? | **1.00** | **1.00** | 0.50 |
| 6 | ¿Cuáles son los documentos requeridos para inscribirse a un programa de pregrado? | **1.00** | 1.00 | **1.00** |
| 7 | ¿Qué porcentaje máximo puede tener un examen final sobre la nota total? | **1.00** | N/A | 0.00 |
| 8 | ¿Cuánto tiempo tiene un estudiante antiguo para solicitar devolución del dinero de matrícula si se retira? | **1.00** | 0.87 | **1.00** |
| 9 | ¿Cuáles son los tipos de práctica profesional reconocidos por la institución? | 0.00 | 0.00 | 0.00 |
| 10 | ¿Cuáles son los requisitos para graduarse en la Fundación Universitaria Konrad Lorenz? | **1.00** | 0.94 | 0.00 |
| | **Promedio General** | **0.90** | **0.68** | **0.43** |

#### Análisis por Métrica

##### Faithfulness (Fidelidad) → Promedio: 0.90
- **Definición:** Mide si la respuesta generada está fundamentada en el contexto recuperado.
- **Resultado:** Excelente. En 9 de 10 preguntas la respuesta fue completamente fiel al contexto recuperado.
- **Anomalía:** La pregunta #9 ("Tipos de práctica profesional") obtuvo **0.00**, lo que indica que el LLM alucinó o no encontró soporte suficiente en el contexto recuperado para su respuesta.

##### Answer Relevancy (Relevancia de la Respuesta) → Promedio: 0.68
- **Definición:** Evalúa si la respuesta es directamente relevante para la pregunta formulada.
- **Resultado:** Moderado. Hay alta variabilidad entre preguntas.
- **Casos problemáticos:**
  - Pregunta #3 ("Perder asignatura por tercera vez") → **0.00**: El sistema no respondió la pregunta específica sobre la tercera pérdida, sino sobre reprobación general.
  - Pregunta #9 ("Tipos de práctica profesional") → **0.00**: Respuesta irrelevante generada.
  - Pregunta #1 ("Inasistencias") → **0.54**: Respuesta parcialmente relevante, mezcló información de inasistencias con sanciones disciplinarias.
- **Casos exitosos:** Preguntas #5, #6 y #4 obtuvieron puntuaciones superiores a 0.95, indicando respuestas altamente relevantes.

##### Context Precision (Precisión del Contexto) → Promedio: 0.43
- **Definición:** Mide qué proporción de los chunks recuperados son realmente relevantes para responder la pregunta.
- **Resultado:** Necesita mejora. Aunque `faithfulness` es alto, el retriever a menudo incluye contexto irrelevante.
- **Casos críticos:**
  - Preguntas #1, #3, #7, #9, #10 obtuvieron **0.00**: El retriever no encontró los artículos correctos o incluyó información no relacionada (ej. sanciones disciplinarias, matrícula, prácticas profesionales).
  - Pregunta #2 obtuvo **0.75**: Recuperó contexto parcialmente relevante pero incluyó artículos sobre inasistencias.

#### Hallazgos Clave

1. **El LLM es conservador:** Cuando encuentra contexto relevante, rara vez alucina (faithfulness 0.90). Sin embargo, cuando el contexto es impreciso, tiende a no responder o a responder de forma genérica.

2. **Problema de recuperación en artículos específicos:** Las preguntas sobre prácticas profesionales, graduación y reingreso obtuvieron `context_precision` de 0.00, lo que sugiere que el embedding no está capturando bien la similitud semántica para esos temas o que los chunks relevantes no están bien diferenciados.

3. **El prompt TFTCR + Few-Shot funciona para respuestas correctas:** Las preguntas con alta `answer_relevancy` (>0.85) corresponden a respuestas estructuradas y directas, lo que valida el diseño del prompt.

#### Recomendaciones

| Prioridad | Acción | Impacto Esperado |
|---|---|---|
| Alta | Revisar el chunking de artículos sobre prácticas profesionales y graduación | Mejorar `context_precision` en temas específicos |
| Alta | Ajustar el `k` del retriever o probar MMR con `fetch_k` mayor | Reducir contexto irrelevante en recuperación |
| Media | Ampliar el dataset de evaluación con más variantes de preguntas | Métricas más robustas y representativas |
| Media | Implementar re-ranking de contextos con un modelo de cross-encoder | Precisión de recuperación más alta |
| Baja | Considerar aumentar `chunk_size` para capturar artículos completos | Mejor coherencia semántica en chunks |

---

## Requisitos e Instalación

### Requisitos Previos

- Python 3.10+
- Windows PowerShell o terminal compatible
- Clave API de Google Gemini (configurada en `.env`)

### Clonar el Repositorio

```powershell
git clone <url-del-repositorio>
cd ChatReglamento
```

### Instalar Dependencias

El proyecto incluye un entorno virtual pre-configurado (`venv/`). Si necesita recrearlo:

```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno
.\env\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Configurar Variables de Entorno

El archivo `.env` debe contener la clave API de Google Gemini:

```env
GOOGLE_API_KEY=tu_clave_api_aqui
```

> **Nota:** El archivo `.env` está incluido en `.gitignore` por seguridad.

---

## Uso

### Paso 1: Ingesta del PDF (ejecutar una vez)

Procesa el documento PDF y crea la base de datos vectorial:

```powershell
.\env\Scripts\python.exe 1_ingesta.py
```

**Fuente del PDF (ya se encuentra en la carpeta):** `pdfs/reglamento.pdf` (588 KB)

Este script:
- Lee el PDF desde `pdfs/reglamento.pdf`
- Segmenta el texto en chunks semánticos
- Genera embeddings multilingües
- Persiste la base vectorial en `chroma_db/`

### Paso 2: Iniciar el Chatbot

Lanza la interfaz web conversacional:

```powershell
.\env\Scripts\python.exe 2_app_gradio.py
```

La aplicación estará disponible en `http://localhost:7860`.

**Funcionalidades de la interfaz:**
- **Chat conversacional:** Escribe tu pregunta y presiona Enter o clic en "Enviar"
- **Mostrar fuentes consultadas:** Activa el checkbox para ver los fragmentos del reglamento que el sistema consultó para generar la respuesta
- **Limpiar conversación:** Botón para reiniciar el chat

### Paso 3 (Opcional): Evaluar el Sistema

Ejecuta la evaluación con RAGAS:

```powershell
.\env\Scripts\python.exe 3_evaluacion_ragas.py
```

Genera el informe `informe_evaluacion_rag.csv`.

### Diagnóstico (Opcional)

Verifica que todas las librerías funcionen correctamente:

```powershell
.\env\Scripts\python.exe diagnostico.py
```

---

## Estructura del Proyecto

```
ChatReglamento/
├─── 1_ingesta.py              # Pipeline Paso 1: Ingesta PDF y creación de Vector DB
├─── 2_app_gradio.py           # Pipeline Paso 2: Interfaz web de chat con Gradio (gr.Blocks + toggle de fuentes)
├─── 3_evaluacion_ragas.py     # Pipeline Paso 3: Evaluación RAGAS con métricas
├─── diagnostico.py            # Script de diagnóstico de librerías
├─── Prueba.py                 # Test mínimo de Streamlit
├─── requirements.txt          # Dependencias Python (~160 paquetes)
├─── .env                      # Variables de entorno (GOOGLE_API_KEY)
├─── .gitignore                # Excluye venv/ y .env
├─── pdfs/
│   └─── reglamento.pdf          # Documento fuente: Reglamento Institucional (588 KB)
├─── chroma_db/                # Base de datos vectorial persistida (ChromaDB)
│   ├─── chroma.sqlite3        # Metadatos de la colección
│   └─── ...                   # Índices HNSW y datos vectoriales
└─── venv/                     # Entorno virtual Python con dependencias instaladas
```

### Descripción de Archivos

| Archivo | Descripción |
|---|---|
| `1_ingesta.py` | Carga el PDF, segmenta en chunks, genera embeddings y persiste en ChromaDB |
| `2_app_gradio.py` | Aplicación principal con interfaz Gradio (gr.Blocks) para chat conversacional con toggle de fuentes |
| `3_evaluacion_ragas.py` | Pipeline de evaluación con 10 preguntas de prueba y métricas RAGAS |
| `diagnostico.py` | Verifica el funcionamiento de librerías críticas (embeddings, ChromaDB, Gemini) |
| `Prueba.py` | Test mínimo de Streamlit (2 líneas) |
| `requirements.txt` | Lista completa de dependencias Python |
| `.env` | Configuración de variables de entorno (API keys) |
| `pdfs/reglamento.pdf` | Documento fuente del Reglamento Institucional |
| `chroma_db/` | Base de datos vectorial persistida con los embeddings del reglamento |

---

## Tecnologías Utilizadas

| Categoría | Tecnologías |
|---|---|
| **LangChain** | langchain, langchain-core, langchain-community, langchain-chroma, langchain-huggingface, langchain-google-genai |
| **Vector DB** | chromadb |
| **Embeddings** | sentence-transformers, torch, transformers |
| **LLM** | google-genai (Gemini) |
| **UI** | gradio, streamlit |
| **Evaluación** | ragas, datasets |
| **PDF** | pypdf |
| **Utilidades** | python-dotenv, pandas, numpy |

---

## Notas Técnicas

### Workarounds para Windows

El script `diagnostico.py` incluye variables de entorno para evitar problemas comunes en Windows:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Formato del Chatbot

El chatbot utiliza el formato de mensajes de Gradio 6.x, donde el historial se estructura como una lista de diccionarios:
```python
[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

### Modelo de Embeddings

El modelo `paraphrase-multilingual-MiniLM-L12-v2` fue seleccionado por:
- Optimización para español y 50+ idiomas
- Tamaño reducido (12 capas, ~118M parámetros)
- Buen balance entre velocidad y calidad semántica
- Compatibilidad con SentenceTransformers

### Configuración del LLM

- **Temperature 0.0:** Garantiza respuestas deterministas y reproducibles, esencial para un sistema de consulta regulatoria donde la consistencia es crítica.
- **Modelo Flash Lite:** Optimizado para velocidad y costo, adecuado para consultas de baja latencia.

---
