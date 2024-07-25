import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain_cohere import ChatCohere
import getpass
import os
from langchain.chains import RetrievalQAWithSourcesChain

st.title('RAG Application')

st.sidebar.header('Enter the Links of Articles🚀🛸 : ')
main_placeholder = st.empty()

urls = []
for i in range(4):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls = st.sidebar.button("Process URLs", type="primary")

if process_urls:
    main_placeholder.text('Loading Text From URLs....✅✅✅')
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    main_placeholder.text('Converting Text into chunks....✅✅✅')
    splitter = RecursiveCharacterTextSplitter(
        separators=['/n/n','/n','.'],
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = splitter.split_documents(data)

    main_placeholder.text('Converting each chunk into Sentence Embedding....✅✅✅')
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vector_index_h = FAISS.from_documents(chunks, embeddings)

    main_placeholder.text('Saving the vectors as pkl file....✅✅✅')
    filepath = "vector_index.pkl"
    with open(filepath,"wb") as f:
        pickle.dump(vector_index_h,f)

    main_placeholder.text('Your chatbot is Ready....✅✅✅ You can start to ask your questions....✅✅✅')


llm = ChatCohere(
    model="command-r-plus",
    temperature=0,
    cohere_api_key="B.............................U",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

query = main_placeholder.text_input("Query : ")
if query:
    filepath = "vector_index.pkl"
    if os.path.exists(filepath):
        with open(filepath,"rb") as f:
            vector_index = pickle.load(f)
    st.subheader('Answer :')
    
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_index.as_retriever())
    results = chain({"question":query},return_only_outputs=True)
    st.write(results['answer'])

    sources = results.get('sources','')
    if sources:
        st.subheader('Sources :')
        l = sources.split('/n')
        for s in l:
            st.write(s)
    

