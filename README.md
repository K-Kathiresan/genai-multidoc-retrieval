## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
"https://openreview.net/pdf?id=hSyW5go0v8",
"https://openreview.net/pdf?id=VTF8yNQM66",
"https://openreview.net/pdf?id=6PmJoRfdaK"
]
papers = [
"selfrag.pdf",
"swebench.pdf",
"longlora.pdf"
]
for url, paper in zip(urls, papers):
    get_ipython().system('wget "{url}" -O "{paper}"')
from utils import get_doc_tools
from pathlib import Path
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt="""You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.""",
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
"Summarize the main idea of Self-RAG and SWE-Bench"
)
print(str(response))
response = agent.query(
"What problem does Self-RAG solve compared to traditional retrieval methods?"
)
print(str(response))
response = agent.query(
"Compare the evaluation method used in SWE-Bench with the approach used in Self-RAG"
)
print(str(response))
```

### OUTPUT:
<img width="989" height="688" alt="image" src="https://github.com/user-attachments/assets/681d4c9b-7b1a-408e-bc27-6acae9273aed" />
<img width="971" height="428" alt="image" src="https://github.com/user-attachments/assets/774f7d08-00fa-4b38-8463-9874d8906bbf" />
<img width="980" height="864" alt="image" src="https://github.com/user-attachments/assets/df473832-30e9-4b1f-9522-9b703de9460e" />


### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    
