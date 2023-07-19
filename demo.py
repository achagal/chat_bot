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
db = FAISS.load_local('U:\Projects\Chat_bot', embeddings, "faiss_index")



# using this site as guidance, create a bot that has memory
# and can return the source documents from where is pulled information from (still need to do this)
# site: https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#make a conversation retrieval chain 
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), db.as_retriever(), memory=memory)

chat_history = []


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










# creates the simple local streamlit interface
# creating local server
st.image("nreca_logo.jpg")
st.title('NRECA Chat Bot')
st.markdown("I am here to answer any questions you have about " 
            "NRECA and point you in the right direction to finding the answers yourself."
            " Hope I can help!")


# creates an input text box and stores question user asks into prompt
prompt = st.text_input('Ask your question here')
if prompt:
    # retrieves the answer from the embeddings
    result = qa({"question": prompt})

    # outputs the recent question
    prompt = split_output(prompt)
    st.text("Question: " + prompt)
    # outputs response to that question in a text box below it
    output = split_output(result["answer"])
    st.text("Answer: " + output)

    # return the source document 

    prompt = ''









# function to save in case we need later
# is the start/skeleton of how we would integrate the source return functionality
# returns what document (the file path) th ebot retrieved its information from
def answer_with_source():
    llm = OpenAI(temperature=0)
    question_generator =LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")


    chain = ConversationalRetrievalChain(
        retriever=db.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )

    chat_history = []
    query = "Does NRECA think electric vehicles will help or hurt electric cooperatives, and why?"
    result = chain({"question": query, "chat_history": chat_history})

    print(result['answer'])




# function to save in case we need later
# now that we have a trained model 
def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Template to use for the system message prompt
    # change this template to match our purposes
    template = """
        You are a helpful assistant that that can answer questions about electric vehicles 
        based on the documents found on NRECA's website: {docs}
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs