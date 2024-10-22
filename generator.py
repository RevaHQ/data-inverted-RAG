from glob import glob
import random
from langchain_ollama import ChatOllama

SAMPLES = 1000
CHUNK_SIZE = 4
MAKE_QUESTION = """You are a customer setting up your Intercom AI agent and are having problems with a specific step. 
You either do not understand the docs or are having technical issues figuring out how to follow them. 
Below are the docs, write a question based on a specific part of the docs that you would ask a customer support agent to get help. Do not summarize the docs, no-one cares.
Pretend to have feelings and a persona, ask only one question, do not refer to the docs, pretend you have not read them.

Docs:
{docs}

You are pretending to be a customer who is having a problem for the purposes of training a customer support rep, ask a question that is answered by these docs.
"""
QUESTION_FORMAT = """You are a customer support agent acting on behalf of Intercom to help customers.

The customer has this question:
{question}

A search of the support docs found:
{docs}

Use this information, if relevant, to offer an answer to the user's query. Be polite and friendly.
"""

files = [x for x in glob("./docs/*.md")]

idx = 0
while idx < SAMPLES:
    fpath = random.choice(files)
    with open(fpath) as fin:
        lines = fin.read().split("\n")
        lines = [x.strip() for x in lines if not x.startswith("![]")]
        docs = [lines[i : i + CHUNK_SIZE] for i in range(0, len(lines), CHUNK_SIZE)]
    lines = random.choice(docs)
    docs = "\n".join(lines)

    llm = ChatOllama(model="llama3.1")
    resp = llm.invoke(MAKE_QUESTION.format(docs=docs))
    question = resp.content

    print("-" * 70)
    print(question)

    with open(f"./prompts/prompt_{idx}.txt", "w") as fout:
        fout.write(QUESTION_FORMAT.format(question=question, docs=docs))
    idx += 1
