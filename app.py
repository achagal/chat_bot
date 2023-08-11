import os
from flask import Flask, request, jsonify
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


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#make a conversation retrieval chain 
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), db.as_retriever(), memory=memory)

chat_history = []


# function to format ouput so that it fits on the screen
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







app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['text']

    # retrieves the answer from the user prompt
    result = qa({"question": user_input})

    # format output
    output = split_output(result["answer"])

    # return output to html file, displaying it on site
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)





# store files in S3 bucket
# launch files on ec2 server

# have IIS enabled
# open fire wall to allow https access

# ask help desk to create a fire wall for a https port for 8501 on my machine

# https://LT74347.va.nreca.org:8501







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