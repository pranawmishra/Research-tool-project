import os
import streamlit as st
import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
import langchain_helper


st.title('News Research tool')
st.sidebar.title('News article URLs')

urls = []

for i in range(3):
    url = st.sidebar.text_input(f'URL{i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URL')
file_path = 'faiss_store_openai.pkl'

main_placeholder = st.empty()
if process_url_clicked:
    # load data
    data = langchain_helper.data_loading(urls,main_placeholder)
    # split data
    docs = langchain_helper.splitting_data(data,main_placeholder)
    # create embeddings and save it to FAISS index
    vectorstore_openai = langchain_helper.creating_embedding(docs,main_placeholder)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input('Question: ')
if query:
    if os.path.exists(file_path):
        answer,result = langchain_helper.provide_answer_and_source(file_path, query, st)
        st.header("Answer")
        st.write(answer)

        # display source if available
        sources_list = langchain_helper.get_source(result,st)
        for source in sources_list:
            st.write(source)


