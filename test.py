import os
import textwrap
import textract
import time
import uuid
import streamlit as st
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# set up environment
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()


#load the completed data base
db = FAISS.load_local('./', embeddings)

ques = "tell me about NRECA's cybersecurity"


def split_output(string, every=10):
    # splits up the text into a list of words 
    words = string.split(' ')
    lines = []
    line = [] 
    for word in words:
        # if we reached the 10 word line maximum, 
        # append this line to the paragraph and create a new line
        if len(line) == every:
            lines.append(' '.join(line))
            line = []
        # add word from this iteration to line
        line.append(word)
    # if the last line didn't have 10 words in it, then we have to manually add it
    if line:
        lines.append(' '.join(line))
    
    # joins all the lines with newline characters seperating them 
    return '\n'.join(lines)



def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    # docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Template to use for the system message prompt
    # change this template to match our purposes
    template = """
        You are a helpful assistant that that can answer questions about the National
        Rural Electric Cooperative Association using information from NRECA's website: {docs}
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed, but also not exceed more than around 200 words.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs[0].page_content)
    response = split_output(response)
    return response


print(get_response_from_query(db, ques))