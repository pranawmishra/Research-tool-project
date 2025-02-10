import os
import streamlit as st
import pickle
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
import langchain_helper

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
embedding = CohereEmbeddings(model="embed-english-light-v3.0")
st.title('General Research tool')
st.sidebar.title(' URLs')

urls = []

for i in range(3):
    url = st.sidebar.text_input(f'URL{i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URL')

main_placeholder = st.empty()
if process_url_clicked:
    # load data
    data = langchain_helper.data_loading(urls,main_placeholder)
    # split data
    docs = langchain_helper.splitting_data(data,main_placeholder)
    # create embeddings and save it to FAISS index
    vectorstore = langchain_helper.creating_embedding(docs,embedding,main_placeholder)
    # Save the FAISS index to a pickle file
    vectorstore.save_local("faiss_index")



query = main_placeholder.text_input('Question: ')
if query:
    st.text("Loading vectorstore......Please Wait......")
    vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    answer = langchain_helper.provide_answer_and_source(vectorstore, query, st)
    st.header("Answer")
    st.write(answer)


