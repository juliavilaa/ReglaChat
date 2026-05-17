# ChatReglamento - Asistente de Consulta del Reglamento Institucional

Sistema RAG (Retrieval-Augmented Generation) que permite realizar consultas en lenguaje natural sobre el Reglamento Institucional de la Fundaci\u00f3n Universitaria Konrad Lorenz, utilizando un documento PDF como fuente de conocimiento y generando respuestas fundamentadas exclusivamente en su contenido.

---

## Tabla de Contenidos

1. [Descripci\u00f3n de la Soluci\u00f3n](#descripci\u00f3n-de-la-soluci\u00f3n)
2. [Arquitectura](#arquitectura)
3. [Flujo del Proceso](#flujo-del-proceso)
4. [Ingesta de Documentos](#ingesta-de-documentos)
5. [Vectorizaci\u00f3n](#vectorizaci\u00f3n)
6. [Construcci\u00f3n del Prompt Aumentado](#construcci\u00f3n-del-prompt-aumentado)
7. [Evaluaci\u00f3n e Informe de Resultados](#evaluaci\u00f3n-e-informe-de-resultados)
8. [Requisitos e Instalaci\u00f3n](#requisitos-e-instalaci\u00f3n)
9. [Uso](#uso)
10. [Estructura del Proyecto](#estructura-del-proyecto)

---

## Descripci\u00f3n de la Soluci\u00f3n

ChatReglamento es un chatbot inteligente basado en la arquitectura RAG que permite a estudiantes, docentes y personal administrativo consultar de forma conversacional el contenido del Reglamento Institucional. El sistema:

- **Ingiere** el documento PDF del reglamento y lo procesa en fragmentos sem\u00e1nticos
- **Vectoriza** cada fragmento usando embeddings multiling\u00fc\u00f1es optimizados para espa\u00f1ol
- **Recupera** los fragmentos m\u00e1s relevantes ante cada consulta del usuario
- **Genera** respuestas precisas utilizando Google Gemini, restringidas \u00fanicamente al contenido del reglamento
- **Eval\u00faa** la calidad del sistema mediante el framework RAGAS con m\u00e9tricas de fidelidad, relevancia y precisi\u00f3n del contexto

---

## Arquitectura

```
\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
\u2502                    ARQUITECTURA RAG                       \u2502
\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
\u2502                                                               \u2502
\u2502  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510   \u2502
\u2502  \u2502  PDF      \u2502\u2500\u2500\u2500>\u2502 PyPDF     \u2502\u2500\u2500\u2500>\u2502 Text Splitter   \u2502   \u2502
\u2502  \u2502 reglamento  \u2502    \u2502 Loader    \u2502    \u2502 (chunking)      \u2502   \u2502
\u2502  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502
\u2502                                              \u2502            \u2502
\u2502                                              v            \u2502
\u2502                                    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510   \u2502
\u2502                                    \u2502 Embeddings    \u2502   \u2502
\u2502                                    \u2502 paraphrase-   \u2502   \u2502
\u2502                                    \u2502 multilingual  \u2502   \u2502
\u2502                                    \u2502 MiniLM-L12    \u2502   \u2502
\u2502                                    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502
\u2502                                             \u2502         \u2502
\u2502                                             v         \u2502
\u2502                                    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510   \u2502
\u2502                                    \u2502  ChromaDB     \u2502   \u2502
\u2502                                    \u2502  (vector DB   \u2502   \u2502
\u2502                                    \u2502   persistida) \u2502   \u2502
\u2502                                    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502
\u2502                                             \u2502         \u2502
\u2502  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u2502         \u2502   \u2502
\u2502  \u2502  Usuario  \u2502\u2500\u2500\u2500>\u2502 Gradio UI \u2502\u2500\u2500\u2500>\u2502 Retriever \u2502   \u2502
\u2502  \u2502 Pregunta  \u2502    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502
\u2502  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518                        \u2502         \u2502   \u2502
\u2502        \u2502                                v         \u2502   \u2502
\u2502        \u2502                        \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510   \u2502   \u2502
\u2502        \u2502                        \u2502 Prompt +    \u2502   \u2502   \u2502
\u2502        \u2502                        \u2502 Contexto    \u2502\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2502   \u2502
\u2502        \u2502                        \u2502 Aumentado   \u2502   \u2502   \u2502
\u2502        \u2502                        \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502   \u2502
\u2502        \u2502                                \u2502         \u2502   \u2502
\u2502        \u2502                                v         \u2502   \u2502
\u2502        \u2502                        \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510   \u2502   \u2502
\u2502        \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500>\u2502 Gemini LLM  \u2502\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2502   \u2502
\u2502                                 \u2502 (respuesta) \u2502   \u2502   \u2502
\u2502                                 \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502   \u2502
\u2502                                             \u2502         \u2502   \u2502
\u2502  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510    \u2502         \u2502   \u2502
\u2502  \u2502 RAGAS     \u2502<\u2500\u2500\u2500\u2502 Evaluaci\u00f3n\u2502<\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502   \u2502
\u2502  \u2502 (m\u00e9tricas) \u2502    \u2502          \u2502                \u2502   \u2502
\u2502  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518                \u2502   \u2502
\u2502                                               \u2502   \u2502
\u2502  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510                                \u2502   \u2502
\u2502  \u2502 Informe   \u2502<\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518   \u2502
\u2502  \u2502 CSV       \u2502                                    \u2502
\u2502  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518                                    \u2502
\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518
```

### Componentes Principales

| Componente | Tecnolog\u00eda | Funci\u00f3n |
|---|---|---|
| **Parser PDF** | PyPDFLoader | Extracci\u00f3n de texto del documento |
| **Text Splitter** | RecursiveCharacterTextSplitter | Segmentaci\u00f3n sem\u00e1ntica por art\u00edculos |
| **Embeddings** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Vectorizaci\u00f3n multiling\u00fce |
| **Vector DB** | ChromaDB | Almacenamiento y b\u00fasqueda por similitud |
| **LLM** | Google Gemini (gemini-3.1-flash-lite-preview) | Generaci\u00f3n de respuestas |
| **UI** | Gradio (gr.Blocks) | Interfaz web conversacional con chatbot y controles |
| **Fuentes** | gr.Checkbox + gr.Textbox | Toggle para mostrar los fragmentos consultados del reglamento |
| **Evaluaci\u00f3n** | RAGAS | M\u00e9tricas de calidad del sistema RAG |
| **Orquestaci\u00f3n** | LangChain | Pipeline RAG unificado |

---

## Flujo del Proceso

El sistema opera en tres etapas secuenciales:

### Etapa 1: Ingesta y Vectorizaci\u00f3n (ejecuci\u00f3n \u00fanica)
1. Lectura del PDF `pdfs/reglamento.pdf`
2. Segmentaci\u00f3n en chunks con preservaci\u00f3n de estructura legal
3. Generaci\u00f3n de embeddings multiling\u00fces
4. Almacenamiento en ChromaDB persistente

### Etapa 2: Consulta Conversacional (servicio continuo)
1. El usuario formula una pregunta en la interfaz Gradio
2. La pregunta se vectoriza y se busca en ChromaDB (top-5 chunks)
3. Se construye el prompt aumentado con el contexto recuperado
4. Gemini genera la respuesta restringida al reglamento
5. La respuesta se muestra al usuario

### Etapa 3: Evaluaci\u00f3n (bajo demanda)
1. Se ejecutan 10 preguntas de prueba con respuestas de referencia
2. RAGAS eval\u00faa faithfulness, answer_relevancy y context_precision
3. Se genera el informe `informe_evaluacion_rag.csv`

---

## Ingesta de Documentos

El proceso de ingesta (`1_ingesta.py`) transforma el documento PDF en una base de conocimiento vectorizada:

### Configuraci\u00f3n de Chunking

| Par\u00e1metro | Valor | Justificaci\u00f3n |
|---|---|---|
| **Chunk Size** | 1200 caracteres | Balance entre contexto y precisi\u00f3n |
| **Chunk Overlap** | 200 caracteres | Preserva contexto entre fragmentos |
| **Separadores** | `["\nARTICULO", "\nPARAGRAFO", "\n\n", "\n", ".", " "]` | Respeta la estructura legal del documento |

### Estrategia de Segmentaci\u00f3n

El `RecursiveCharacterTextSplitter` aplica los separadores en orden de prioridad:
1. Intenta dividir en l\u00edmites de `\nARTICULO` (preserva art\u00edculos completos)
2. Si no cabe, usa `\nPARAGRAFO` (preserva par\u00e1grafos)
3. Falls back a separadores gen\u00e9ricos (`\n\n`, `\n`, `.`, ` `)

Esto asegura que cada chunk mantenga coherencia sem\u00e1ntica dentro del contexto regulatorio.

---

## Vectorizaci\u00f3n

### Modelo de Embeddings

- **Modelo:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Idiomas:** Optimizado para 50+ idiomas incluyendo espa\u00f1ol
- **Dimensi\u00f3n:** 384 dimensiones por vector
- **Backend:** PyTorch con inferencia local

### Proceso

```
Texto del chunk \u2500\u2500> Tokenizaci\u00f3n \u2500\u2500> Modelo Transformer \u2500\u2500> Vector 384d \u2500\u2500> ChromaDB
```

Cada fragmento de texto se convierte en un vector denso que captura su significado sem\u00e1ntico, permitiendo b\u00fasquedas por similitud coseno durante la recuperaci\u00f3n.

### Base de Datos Vectorial

- **Motor:** ChromaDB con persistencia en disco
- **Ubicaci\u00f3n:** `./chroma_db/`
- **Colecci\u00f3n:** `reglamento_db`
- **\u00cdndice:** HNSW (Hierarchical Navigable Small World) para b\u00fasqueda eficiente
- **M\u00e9trica:** Distancia coseno (L2)

---

## Construcci\u00f3n del Prompt Aumentado

El prompt aumentado combina la pregunta del usuario con el contexto recuperado de la base vectorial:

### Template del Prompt (TFTCR + Few-Shot)

```
Eres un asistente institucional experto.

[TASK] Tu objetivo es responder consultas de estudiantes basandote UNICAMENTE en el reglamento institucional proporcionado.
[FORMAT] Responde de manera clara y directa. Usa viñetas o listas numeradas si la respuesta implica multiples pasos o condiciones.
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

| Elemento | Descripci\u00f3n |
|---|---|
| **Instrucci\u00f3n de restricci\u00f3n** | Obliga al LLM a usar solo el contexto proporcionado |
| **Fallback** | Instruye responder "No encuentro esa informaci\u00f3n" si no hay datos suficientes |
| **{context}** | Los 5 chunks m\u00e1s relevantes recuperados de ChromaDB |
| **{question}** | La pregunta formulada por el usuario |

### Par\u00e1metros del LLM

| Par\u00e1metro | Valor | Prop\u00f3sito |
|---|---|---|
| **Modelo** | `gemini-3.1-flash-lite-preview` | Balance velocidad/calidad |
| **Temperature** | `0.0` | Respuestas deterministas y consistentes |
| **k (retriever)** | `5` | Top-5 chunks m\u00e1s relevantes (equilibrio cobertura/precisi\u00f3n) |
| **Tipo de b\u00fasqueda** | `similarity` | Similitud coseno en el espacio vectorial |
| **max_output_tokens** | `2048` | Respuestas completas sin truncamiento |

### Flujo de Construcci\u00f3n

```
1. Usuario env\u00eda pregunta \u2192 "?\u00bfCu\u00e1l es la nota m\u00ednima para aprobar?"
                                    \u2193
2. Pregunta se vectoriza \u2192 Embedding de la consulta
                                    \u2193
3. B\u00fasqueda en ChromaDB \u2192 Top-6 chunks m\u00e1s similares
                                    \u2193
4. Concatenaci\u00f3n de contexto \u2192 Todos los chunks unidos en un solo string
                                    \u2193
5. Inyecci\u00f3n en el template \u2192 Prompt completo con contexto + pregunta
                                    \u2193
6. Env\u00edo a Gemini \u2192 Generaci\u00f3n de respuesta fundamentada
                                    \u2193
7. Respuesta al usuario \u2192 "La nota m\u00ednima para aprobar es 30 sobre 50..."
```

---

## Evaluaci\u00f3n e Informe de Resultados

El sistema incluye un pipeline de evaluaci\u00f3n basado en el framework **RAGAS** que mide la calidad del sistema RAG con tres m\u00e9tricas fundamentales:

### M\u00e9tricas Evaluadas

| M\u00e9trica | Descripci\u00f3n | Rango |
|---|---|---|
| **Faithfulness** | \u00bfLa respuesta est\u00e1 fundamentada en el contexto recuperado? | 0.0 - 1.0 |
| **Answer Relevancy** | \u00bfLa respuesta es relevante para la pregunta? | 0.0 - 1.0 |
| **Context Precision** | \u00bfLos contextos recuperados son relevantes para la pregunta? | 0.0 - 1.0 |

### Dataset de Evaluaci\u00f3n

El archivo `3_evaluacion_ragas.py` contiene 10 preguntas de prueba con respuestas de referencia que cubren temas clave del reglamento:

| # | Tema | Pregunta |
|---|---|---|
| 1 | Inasistencias | \u00bfCu\u00e1l es el porcentaje de inasistencias que puede causar la p\u00e9rdida del curso? |
| 2 | Nota m\u00ednima | \u00bfCu\u00e1l es la nota m\u00ednima para aprobar un curso? |
| 3 | Reprobaci\u00f3n | \u00bfQu\u00e9 pasa si un estudiante reprueba tres veces un curso? |
| 4 | Horas acad\u00e9micas | \u00bfCu\u00e1ntas horas acad\u00e9micas tiene un cr\u00e9dito? |
| 5 | Menci\u00f3n honor\u00edfica | \u00bfCu\u00e1l es el promedio para menci\u00f3n honor\u00edfica? |
| 6 | Matr\u00edcula pregrado | \u00bfQu\u00e9 documentos se requieren para la matr\u00edcula de pregrado? |
| 7 | Peso examen final | \u00bfCu\u00e1l es el porcentaje m\u00e1ximo que puede valer un examen final? |
| 8 | Retiro y reembolso | \u00bfEn cu\u00e1ntos d\u00edas se puede pedir reembolso al retirarse? |
| 9 | Pr\u00e1cticas profesionales | \u00bfCu\u00e1ntos tipos de pr\u00e1ctica profesional existen? |
| 10 | Graduaci\u00f3n | \u00bfCu\u00e1les son los requisitos para graduarse? |

### LLM como Juez

RAGAS utiliza Gemini como juez de evaluaci\u00f3n mediante `LangchainLLMWrapper`, comparando las respuestas generadas contra las respuestas de referencia y el contexto recuperado.

### Generaci\u00f3n del Informe

Los resultados se exportan a `informe_evaluacion_rag.csv` con las siguientes columnas:
- `question`: Pregunta de prueba
- `answer`: Respuesta generada por el sistema
- `contexts`: Contextos recuperados
- `ground_truth`: Respuesta de referencia
- `faithfulness`: Puntuaci\u00f3n de fidelidad
- `answer_relevancy`: Puntuaci\u00f3n de relevancia
- `context_precision`: Puntuaci\u00f3n de precisi\u00f3n del contexto

### Ejecuci\u00f3n de la Evaluaci\u00f3n

```powershell
.\venv\Scripts\python.exe 3_evaluacion_ragas.py
```

---

## Requisitos e Instalaci\u00f3n

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
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Configurar Variables de Entorno

El archivo `.env` debe contener la clave API de Google Gemini:

```env
GOOGLE_API_KEY=tu_clave_api_aqui
```

> **Nota:** El archivo `.env` est\u00e1 incluido en `.gitignore` por seguridad.

---

## Uso

### Paso 1: Ingesta del PDF (ejecutar una vez)

Procesa el documento PDF y crea la base de datos vectorial:

```powershell
.\venv\Scripts\python.exe 1_ingesta.py
```

**Fuente del PDF:** `pdfs/reglamento.pdf` (588 KB)

Este script:
- Lee el PDF desde `pdfs/reglamento.pdf`
- Segmenta el texto en chunks sem\u00e1nticos
- Genera embeddings multiling\u00fces
- Persiste la base vectorial en `chroma_db/`

### Paso 2: Iniciar el Chatbot

Lanza la interfaz web conversacional:

```powershell
.\venv\Scripts\python.exe 2_app_gradio.py
```

La aplicaci\u00f3n estar\u00e1 disponible en `http://localhost:7860`.

**Funcionalidades de la interfaz:**
- **Chat conversacional:** Escribe tu pregunta y presiona Enter o clic en "Enviar"
- **Mostrar fuentes consultadas:** Activa el checkbox para ver los fragmentos del reglamento que el sistema consult\u00f3 para generar la respuesta
- **Limpiar conversaci\u00f3n:** Bot\u00f3n para reiniciar el chat

### Paso 3 (Opcional): Evaluar el Sistema

Ejecuta la evaluaci\u00f3n con RAGAS:

```powershell
.\venv\Scripts\python.exe 3_evaluacion_ragas.py
```

Genera el informe `informe_evaluacion_rag.csv`.

### Diagn\u00f3stico (Opcional)

Verifica que todas las librer\u00edas funcionen correctamente:

```powershell
.\venv\Scripts\python.exe diagnostico.py
```

---

## Estructura del Proyecto

```
ChatReglamento/
\u251c\u2500\u2500\u2500 1_ingesta.py              # Pipeline Paso 1: Ingesta PDF y creaci\u00f3n de Vector DB
\u251c\u2500\u2500\u2500 2_app_gradio.py           # Pipeline Paso 2: Interfaz web de chat con Gradio (gr.Blocks + toggle de fuentes)
\u251c\u2500\u2500\u2500 3_evaluacion_ragas.py     # Pipeline Paso 3: Evaluaci\u00f3n RAGAS con m\u00e9tricas
\u251c\u2500\u2500\u2500 diagnostico.py            # Script de diagn\u00f3stico de librer\u00edas
\u251c\u2500\u2500\u2500 Prueba.py                 # Test m\u00ednimo de Streamlit
\u251c\u2500\u2500\u2500 requirements.txt          # Dependencias Python (~160 paquetes)
\u251c\u2500\u2500\u2500 .env                      # Variables de entorno (GOOGLE_API_KEY)
\u251c\u2500\u2500\u2500 .gitignore                # Excluye venv/ y .env
\u251c\u2500\u2500\u2500 pdfs/
\u2502   \u2514\u2500\u2500\u2500 reglamento.pdf          # Documento fuente: Reglamento Institucional (588 KB)
\u251c\u2500\u2500\u2500 chroma_db/                # Base de datos vectorial persistida (ChromaDB)
\u2502   \u251c\u2500\u2500\u2500 chroma.sqlite3        # Metadatos de la colecci\u00f3n
\u2502   \u2514\u2500\u2500\u2500 ...                   # \u00cdndices HNSW y datos vectoriales
\u2514\u2500\u2500\u2500 venv/                     # Entorno virtual Python con dependencias instaladas
```

### Descripci\u00f3n de Archivos

| Archivo | Descripci\u00f3n |
|---|---|
| `1_ingesta.py` | Carga el PDF, segmenta en chunks, genera embeddings y persiste en ChromaDB |
| `2_app_gradio.py` | Aplicaci\u00f3n principal con interfaz Gradio (gr.Blocks) para chat conversacional con toggle de fuentes |
| `3_evaluacion_ragas.py` | Pipeline de evaluaci\u00f3n con 10 preguntas de prueba y m\u00e9tricas RAGAS |
| `diagnostico.py` | Verifica el funcionamiento de librer\u00edas cr\u00edticas (embeddings, ChromaDB, Gemini) |
| `Prueba.py` | Test m\u00ednimo de Streamlit (2 l\u00edneas) |
| `requirements.txt` | Lista completa de dependencias Python |
| `.env` | Configuraci\u00f3n de variables de entorno (API keys) |
| `pdfs/reglamento.pdf` | Documento fuente del Reglamento Institucional |
| `chroma_db/` | Base de datos vectorial persistida con los embeddings del reglamento |

---

## Tecnolog\u00edas Utilizadas

| Categor\u00eda | Tecnolog\u00edas |
|---|---|
| **LangChain** | langchain, langchain-core, langchain-community, langchain-chroma, langchain-huggingface, langchain-google-genai |
| **Vector DB** | chromadb |
| **Embeddings** | sentence-transformers, torch, transformers |
| **LLM** | google-genai (Gemini) |
| **UI** | gradio, streamlit |
| **Evaluaci\u00f3n** | ragas, datasets |
| **PDF** | pypdf |
| **Utilidades** | python-dotenv, pandas, numpy |

---

## Notas T\u00e9cnicas

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
- Optimizaci\u00f3n para espa\u00f1ol y 50+ idiomas
- Tama\u00f1o reducido (12 capas, ~118M par\u00e1metros)
- Buen balance entre velocidad y calidad sem\u00e1ntica
- Compatibilidad con SentenceTransformers

### Configuraci\u00f3n del LLM

- **Temperature 0.0:** Garantiza respuestas deterministas y reproducibles, esencial para un sistema de consulta regulatoria donde la consistencia es cr\u00edtica.
- **Modelo Flash Lite:** Optimizado para velocidad y costo, adecuado para consultas de baja latencia.

---

## Licencia y Uso

Este proyecto fue desarrollado como un asistente de consulta del Reglamento Institucional de la Fundaci\u00f3n Universitaria Konrad Lorenz. Las respuestas generadas son orientativas y deben verificarse contra el documento oficial del reglamento.
