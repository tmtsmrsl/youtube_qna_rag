import chainlit as cl
from dotenv import load_dotenv
from utils import load_llm, load_vector_store, create_qa_chain
import os

@cl.on_chat_start
async def on_chat_start():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
    
    vector_store = load_vector_store(pinecone_api_key, local_embeddings=True)
    llm = load_llm(fireworks_api_key)
    chain = create_qa_chain(llm, vector_store)
    
    msg = cl.Message(content="Please ask me a question about machine learning!")
    await msg.send()
    
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 

    response = chain.invoke({"input":message.content})
    answer_src = response["answer"]
    answer = answer_src.answer
    source_documents = answer_src.sources
    formatted_sources = "\n".join(source_documents)
    
    if source_documents:
        elements = [
            cl.Text(name="Sources", content=formatted_sources, display="inline"),
            cl.Video(url=source_documents[0], display="inline"),
        ]
    else:
        elements = []

    await cl.Message(content=answer, elements=elements).send() 