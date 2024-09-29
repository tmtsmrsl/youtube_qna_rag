import os

import chainlit as cl
from dotenv import load_dotenv

from utils import create_qa_chain, load_llm, load_vector_store


@cl.on_chat_start
async def on_chat_start():
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
    
    vector_store = load_vector_store(PINECONE_API_KEY, DEEPINFRA_API_KEY)
    llm = load_llm(GROQ_API_KEY)
    chain = create_qa_chain(llm, vector_store)
    
    msg = cl.Message(content="Welcome to the StatQuest ML Video Search Assistant! Ask me any questions about machine learning, and I’ll help you find relevant StatQuest videos along with their timestamps.")
    await msg.send()
    
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 

    response = chain.invoke({"input":message.content})
    answer_and_src = response["answer"]
    answer = answer_and_src.answer
    source_documents = answer_and_src.sources
    formatted_sources = "\n".join(source_documents)
    
    if source_documents:
        elements = [
            cl.Video(url=source_documents[0], display="inline"),
            cl.Text(name="Sources", content=formatted_sources, display="inline")
        ]
    else:
        elements = []

    await cl.Message(content=answer, elements=elements).send() 