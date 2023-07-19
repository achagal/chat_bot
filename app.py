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

