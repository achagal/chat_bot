import os
import textwrap
import re 
from pdfminer.high_level import extract_text
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


os.environ["OPENAI_API_KEY"] = "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"
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

# reformat this function to take in a text/word document
# likely the same code, just don't need the first lines and
# instead of creatnig a new database each time, just add this
# new information to the data base
def create_db_from_pdfs():

    # change later on
    # only way I could figure out how to initialize a FAISS data base
    # is by creating it from a list[Documents], so I used random pdf from data
    loaderInit = PDFMinerLoader("U:\Projects\Chat_bot\data\White-paper-Series-1.pdf")
    init = loaderInit.load()
    db = FAISS.from_documents(init, embeddings)


    pdf_directory = "U:\Projects\Chat_bot\data"

    for pdf_file in os.listdir(pdf_directory):
        if not pdf_file.endswith(".pdf"):
            continue

        # add back slash and pdf_file name to pdf_directory to complete
        # the path to the current file at this iteration
        file_path = pdf_directory + '\\' + pdf_file

        # extracts text from pdf
        loader = PDFMinerLoader(file_path)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        data = text_splitter.split_documents(doc)


        #text = extract_text(file_path)
    
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        #doc = text_splitter.split_text(text)
        #db = FAISS.aadd_texts(doc, embeddings)
        
        # saves the extracted text as a txt
        
        #text_file = file_path[:len(file_path) - 3] + ('txt')
        #with open(text_file, 'w', encoding='utf-8') as file:
            #file.write(text)
        

        # at the end of each iteration, add the newly formatted file to the data base
        db.add_documents(data)
    
    #end of for loop

    return db

# create a db and fill it using function above
db = create_db_from_pdfs()
















# load the model we already made to further train it
db = FAISS.load_local('U:\Projects\Chat_bot', embeddings, "faiss_index")



# save the data base for future use
db.save_local("U:\Projects\Chat_bot", "faiss_index")

print("Successfuly trained and saved")