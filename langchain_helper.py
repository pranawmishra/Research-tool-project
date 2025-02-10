import pickle
import time
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
str_output_parser = StrOutputParser()
qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """you are helpful assistant who help user in answering questions that is asked to you from provided context. \
                            Your answer should not be greater than 2 lines. Keep your answer very short and do not include any unnecessary lines.\
                            You have to use the provided document to answer the question:\
                            {document}\
                        """),
            ("human", "{input}"),
        ])

def data_loading(urls,mp):
    loader = UnstructuredURLLoader(urls=urls)
    mp.text('Data Loading.....Please wait......')
    data = loader.load()
    return data

def splitting_data(data,mp):
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','],
                                                   chunk_size=1000)
    mp.text('Data Splitting.....Please wait......')
    docs = text_splitter.split_documents(data)
    return docs

def creating_embedding(docs,embedding,mp):
    vectorstore = FAISS.from_documents(docs,embedding)
    mp.text('Embedding vector started building.....Please wait......')
    time.sleep(2)
    return vectorstore

def provide_answer_and_source(vectorstore,query,st):
    # with open(file_path, "rb") as f:
    # vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()
    rag_chain = qa_prompt | llm | str_output_parser
    result = rag_chain.invoke({"input":query,
                                "document":retriever.invoke(query)})
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    return result

        # Display sources, if available
def get_source(result,st):
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        return sources_list


