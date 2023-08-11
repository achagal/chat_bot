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


# set up openAI environment
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()


#load the completed data base
db = FAISS.load_local('./', embeddings)


# function that reformats output to fit on web page
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




# function that uses openAI to answer question
def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    # docs contains a list of embedded text documents, ordered by increasing relevance
    # docs[0] is the most relevant piece of text to the question (query)
    docs = db.similarity_search(query, k=k)

    # sets model to be gpt-3.5
    # can change to any model you want; just copy name from these options:
    # https://platform.openai.com/docs/models/overview
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Template to use for the system message prompt
    # change this template to match our purposes
    template = """
        You are a helpful assistant that that can answer questions about the National
        Rural Electric Cooperative Association using information from NRECA's website: {docs}
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed, but also not exceed more than around 100 words.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    # formats the question to decrease GPT confusion
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # creates a conversation chain, so the model has memory 
    # allows for a user to ask a question, then say "can you expand on THAT"
    # rather than restate the entire question 
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    # docs[0] contains most relevant text, so have GPT answer the question
    # with docs[0] as context. this prevents the Token limit from being exceeded
    # since we are filtering only the most important information
    response = chain.run(question=query, docs=docs[0].page_content)
    response = split_output(response)
    return response



# START OF FRONT END

# hides the stop button so users can't accidentally terminate the program
st.markdown("""
    <style>
    div[data-testid="stToolbar"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        div[data-testid="stTooltip"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

# creates the simple streamlit interface
st.image("nreca_logo.jpg")
st.title('NRECA Chat Bot')
st.markdown("I am here to answer any questions you have about " 
            "NRECA and point you in the right direction to finding the answers yourself."
            " Hope I can help!")


# creates an input text box and stores question user asks into prompt
prompt = st.text_input('Ask your question here')
submit_button = st.button('Submit Question')


if submit_button:
    # starts loop of requesting user input
    if prompt:
        # retrieves the answer from GPT
        result = get_response_from_query(db, prompt)

        # outputs the recent question
        prompt = split_output(prompt)
        st.text("Question: " + prompt)

        # outputs response to that question in a text box below it
        st.text("Answer: " + result)

        # reset prompt so the answer portion clears
        # helps with user interface
        prompt = ''