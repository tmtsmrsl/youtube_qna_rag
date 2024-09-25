from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_community.embeddings import DeepInfraEmbeddings
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import torch

def load_vector_store(pinecone_api_key, deepinfra_api_key=None, local_embeddings=False):
    """
    Load the Pinecone vector store containing our transcript embeddings.
    """
    index_name =  "statquest-machinelearning"
    model_name= "BAAI/bge-base-en-v1.5"
    
    
    if local_embeddings:
        device = 0 if torch.cuda.is_available() else None
        embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                    model_kwargs={"trust_remote_code":True, 
                                                "device":device})
    else:
        if deepinfra_api_key is None:
            raise ValueError("deepinfra_api_key is required if you are not using local embeddings")
        embeddings = DeepInfraEmbeddings(model_id=model_name, 
                                         deepinfra_api_token=deepinfra_api_key)
    
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings,
                                       pinecone_api_key=pinecone_api_key)
    return vector_store


def load_llm(groq_api_key):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature = 0, max_tokens=500, max_retries=5, api_key=groq_api_key)
    return llm


def format_query(query):
    """
    Add a query instruction to the beginning of the search query to slightly 
    improve retrieval performance
    """
    query_instruction = "Represent this sentence for searching relevant passages: "
    return query_instruction + query


def format_docs(docs):
    """
    Format the returned documents before injecting them into the prompt, 
    ensuring they match the structure of the few-shot examples
    """
    formatted_docs = ""
    for doc in docs:
        formatted_docs += f"{doc.page_content}\n"
        id = doc.metadata['video_id']
        start_time = doc.metadata['start_time']
        formatted_docs += f"source: https://www.youtube.com/watch?v={id}&t={round(start_time)}\n\n"
    formatted_docs = formatted_docs.rstrip()
    return formatted_docs


def load_prompt_template():
    prompt_template = '''Given a QUESTION, use the given content to formulate a FINAL_ANSWER (list) with SOURCES (string).
If the answer cannot be found from the given content, just say that you don't know in the FINAL_ANSWER and return an empty list for SOURCES.
Only include sources that are used to formulate the FINAL_ANSWER.

Below are two examples.
QUESTION: What is a stump in ada boost?
=========
Gradient Boost Part 1 (of 4) Regression Main Ideas:
So let's briefly compare and contrast AdaBoost and gradient boost. If we want to use these measurements to predict weight, then AdaBoost starts by building a very short tree called a stump from the training data. And then the amount of say that the new stump has on the final output is based on how well it compensated for those previous errors. Then Adaboost builds the next stump based on errors that the previous stump made. In this example, the new stump did a poor job compensating for the previous stump's errors and its size reflects its reduced amount of say. Then Adaboost builds another stump based on the errors made by the previous stump. And this stump did a little better than the last stump, so it's a little larger.
source: https://www.youtube.com/watch?v=3CC4N4z3GJc&t=174

AdaBoost, Clearly Explained:
And these are the amounts of say for these stumps. Now we add up the amounts of say for this group of stumps and for this group of stumps. Ultimately, the patient is classified as has heart disease because this is the larger sum. Triple bam. To review, the three ideas behind Adaboost are one, Adaboost combines a lot of weak learners to make classifications. The weak learners are almost always stumps. Two, some stumps get more say in the classification than others. And 3. Each stump is made by taking the previous stump's mistakes into account. If we have a weighted genie function, then we use it with the sample weights. Otherwise, we use the sample weights to make a new dataset that reflects those weights. Hooray! We've made it to the end of another exciting StatQuest.
source: https://www.youtube.com/watch?v=LsK-xG1cLYA&t=1167

AdaBoost, Clearly Explained:
To review, the three ideas behind Adaboost are 1. Adaboost combines a lot of weak learners to make classifications. The weak learners are almost always stumps. 2. Some stumps get more say in the classification than others. 3. Each stump is made by taking the previous stump's mistakes into account. BAM! Now let's dive into the nitty-gritty detail of how to create a forest of stumps using Adaboost. First, we'll start with some data. We create a forest of stumps with Adaboost to predict if a patient has heart disease. We will make these predictions based on a patient's chest pain and blocked artery status and their weight. The first thing we do is give each sample a weight that indicates how important it is to be correctly classified.
source: https://www.youtube.com/watch?v=LsK-xG1cLYA&t=210

AdaBoost, Clearly Explained:
Stumps are not great at making accurate classifications. For example, if we were using this data to determine if someone had heart disease or not, Then a full-size decision tree would take advantage of all four variables that we measured, chest pain, blood circulation, blocked arteries, and weight, to make a decision. But a stump can only use one variable to make a decision. Thus, stumps are technically weak learners. However, that's the way Adaboost likes it, and it's one of the reasons why they are so commonly combined. Now back to the random forest. In a random forest, each tree has an equal vote on the final classification. This tree's vote is worth just as much as this tree's vote or this tree's vote.
source: https://www.youtube.com/watch?v=LsK-xG1cLYA&t=101

Gradient Boost Part 1 (of 4) Regression Main Ideas:
And this stump did a little better than the last stump, so it's a little larger. Then Adaboost continues to make stumps in this fashion until it has made the number of stumps you asked for or it has a perfect fit. In contrast, Gradient Boost starts by making a single leaf instead of a tree or stump. This leaf represents an initial guess for the weight for all of the samples. When trying to predict a continuous value like weight, the first guess is the average value. Then Gradient Boost builds a tree. Like Ataboost, this tree is based on the errors made by the previous tree. But unlike Ataboost, this tree is usually larger than a stump. That said, Gradient Boost still restricts the size of the tree.
source: https://www.youtube.com/watch?v=3CC4N4z3GJc&t=222
=========
FINAL_ANSWER: A stump in AdaBoost is a weak learner, typically a very short tree that uses only one variable to make a decision. Stumps are not great at making accurate classifications on their own, but they are combined in AdaBoost to make a stronger classifier. Each stump is made by taking the previous stump's mistakes into account, and some stumps get more say in the final classification than others.
SOURCES: [https://www.youtube.com/watch?v=LsK-xG1cLYA&t=1167, https://www.youtube.com/watch?v=LsK-xG1cLYA&t=210, https://www.youtube.com/watch?v=LsK-xG1cLYA&t=101]

QUESTION: Why is UMAP better than PCA?
=========
StatQuest Linear Discriminant Analysis (LDA) clearly explained.:
It was just looking for the genes with the most variation. So we've seen the differences between LDA and PCA, but now let's talk about some of the similarities. The first similarity is that both methods rank the new axes that they create in order importance. PC1, the first new access that PCA creates, accounts for the most variation in the data. Likewise, PC2, the second new access, does the second best job, and this goes on and on for the number of axes that are created from the data. LD1, the first new access that LDA creates, accounts for the most variation between the categories. LD2, the second new axis, does the second best job, etc., etc., etc. Also, both methods let you dig in and see which genes are driving the new axes. In PCA, this means looking at the loading scores.
source: https://www.youtube.com/watch?v=azXCzI57Yfc&t=814

StatQuest Linear Discriminant Analysis (LDA) clearly explained.:
It's way better than drawing a 10,000 dimension figure that we can't even imagine what it would look like. Here's an example using real data. I'm trying to separate three categories and I've got 10,000 genes. Plotting the raw data would require 10,000 axes. We used LDA to reduce the number to 2. And although the separation isn't perfect, it is still easy to see three separate categories. Now let's use that same dataset to compare LDA to PCA. Here's the LDA plot that we saw before. And now we've applied PCA to the exact same set of genes. PCA doesn't separate the categories nearly as well. We can see lots of overlap between the black and the blue points. However, PCA wasn't even trying to separate those categories. It was just looking for the genes with the most variation.
source: https://www.youtube.com/watch?v=azXCzI57Yfc&t=760

StatQuest Linear Discriminant Analysis (LDA) clearly explained.:
We ran into this same problem when we talked about principal component analysis, or PCA. And if you don't know about principal component analysis, be sure to check out the StatQuest on that subject. It's got a lot of likes and it's helped a lot of people understand how it works and what it does. PCA, if you can remember about it, reduces dimensions by focusing on genes with the most variation. This is incredibly useful when you're plotting data with a lot of dimensions or a lot of genes onto a simple x-y plot. However, in this case, we're not super interested in the genes with the most variation. Instead, we're interested in maximizing the separability between the two groups so that we can make the best decisions. Linear Discriminant Analysis is like PCA. It reduces dimensions.
source: https://www.youtube.com/watch?v=azXCzI57Yfc&t=252

StatQuest PCA main ideas in only 5 minutes!!!:
StatQuest is the best, if you don't think so, then we have different opinions. Hello, I'm Josh Starmer and welcome to StatQuest. Today we're going to be talking about the main ideas behind principal component analysis, and we're going to cover those concepts in five minutes. If you want more details than you get here, be sure to check out my other PCA video. Let's say we had some normal cells. Psst! If you're not a biologist, imagine that these could be people, or cars, or cities, or etc. They could be anything. Even though they look the same, we suspect that there are differences. These might be one type of cell, or one type of person, or car, or city, etc. These might be another type of cell. And lastly, these might be a third type of cell.
source: https://www.youtube.com/watch?v=HMOI_lkzW08&t=0
=========
FINAL_ANSWER: I don't know.
SOURCES: []

Now you are going to answer the following question.
QUESTION: {input}
=========
{context}
=========
FINAL_ANSWER:
SOURCES:
'''
    return prompt_template


class AnswerWithSources(BaseModel):
    """An answer to the question, with sources."""
    
    answer: str = Field(description="The answer to the question if any")
    sources: List[str] = Field(
        description="List of sources (video_url) used to answer the question if any")

def create_qa_chain(llm, vector_store):
    prompt_template = load_prompt_template()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", prompt_template),
        ]
    )
    
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],
            "context": lambda x: format_docs(x["context"]),
        }
        | prompt
        | llm.with_structured_output(AnswerWithSources)
    )
    
    retrieve_docs = (lambda x: format_query(x["input"])) | vector_store.as_retriever(search_kwargs={"namespace": "transcript"})

    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    return chain