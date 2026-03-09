#!/usr/bin/env python
# coding: utf-8

# In[2]:


from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()


# In[3]:


import nest_asyncio
nest_asyncio.apply()


# In[4]:


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


# In[5]:


for url, paper in zip(urls, papers):
    get_ipython().system('wget "{url}" -O "{paper}"')


# In[6]:


from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}

for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]


# In[7]:


all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


# In[12]:


from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")


# In[13]:


from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)


# In[14]:


obj_retriever = obj_index.as_retriever(similarity_top_k=3)


# In[15]:


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


# In[16]:


response = agent.query(
"Summarize the main idea of Self-RAG and SWE-Bench"
)

print(str(response))


# In[17]:


response = agent.query(
"What problem does Self-RAG solve compared to traditional retrieval methods?"
)

print(str(response))


# In[18]:


response = agent.query(
"Compare the evaluation method used in SWE-Bench with the approach used in Self-RAG"
)

print(str(response))

