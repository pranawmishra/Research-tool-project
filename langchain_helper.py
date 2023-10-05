import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
llm = OpenAI(temperature=0.9, max_tokens=500)

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

def creating_embedding(docs,mp):
    embedding = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs,embedding)
    mp.text('Embedding vector started building.....Please wait......')
    time.sleep(2)
    return vectorstore_openai

def provide_answer_and_source(file_path,query,st):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        return result["answer"], result

        # Display sources, if available
def get_source(result,st):
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        return sources_list


