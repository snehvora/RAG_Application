# RAG Application

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain that allows users to input URLs of articles, process these URLs to extract text, and then create a chatbot that can answer questions based on the processed text.

## Features

- **URL Input**: Users can input up to 4 URLs through the Streamlit sidebar.
- **Text Extraction**: Extracts and processes text from the provided URLs.
- **Chunking**: Splits the extracted text into manageable chunks.
- **Embedding**: Converts text chunks into sentence embeddings using a pre-trained model.
- **Vector Indexing**: Creates and saves a vector index of the embeddings.
- **Chatbot**: Provides a chatbot interface for users to ask questions based on the processed text.
- **Source Attribution**: Returns the sources of the information used in the chatbot's answers.

## Installation

1. **Clone the Repository**

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

### Prerequisites

This project uses Poetry for dependency management. Follow the steps below to set up the project.

Ensure you have Poetry installed. If not, you can install it using the following command:

```sh
pip3 install poetry
```
Then, install the dependencies:

```sh
poetry install
```

## Usage

1. **Run the Streamlit App**
   
```sh
poetry run streamlit run app.py
```

2. **Enter URLs**

In the Streamlit sidebar, enter up to 4 URLs of articles you want to process.

4. **Process URLs**

Click the "Process URLs" button to load and process the text from the entered URLs. The app will:

- Extract text from the URLs.
- Split the text into chunks.
- Convert each chunk into sentence embeddings.
- Save the embeddings into a vector index.
  
4. **Ask Questions**
   
Enter your query in the text input box, and the chatbot will provide an answer based on the processed text. The sources of the information will also be displayed.


## Code Explanation

**Streamlit Interface**

```python
st.title('RAG Application')
st.sidebar.header('Enter the Links of Articles🚀🛸 : ')
main_placeholder = st.empty()
```
- Sets up the main title and sidebar header for the Streamlit app.
- main_placeholder is used to display messages during processing.
  
**URL Input and Processing**
```python
urls = []
for i in range(4):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls = st.sidebar.button("Process URLs", type="primary")
```
- Allows users to input up to 4 URLs via the sidebar.
- process_urls button initiates the processing of these URLs.
  
**Text Extraction and Chunking**
```python
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
```
- Uses UnstructuredURLLoader to extract text from the provided URLs.
- Splits the extracted text into chunks using RecursiveCharacterTextSplitter.

**Embedding and Vector Indexing**
```python
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
```
- Converts text chunks into embeddings using the HuggingFace BGE model.
- Creates a FAISS vector index from the embeddings and saves it as a pickle file.
  
**Chatbot Interface**
```python
llm = ChatCohere(
    model="command-r-plus",
    temperature=0,
    cohere_api_key="BVCDr0O3HySj55efzANnVtavYeEAkn5tZUr6wfyU",
    max_tokens=None,
    timeout=None,
    max_retries=2,
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
```
- Sets up a chatbot using ChatCohere for the language model.
- Handles user queries, retrieves answers from the vector index, and displays the results along with sources.

  
**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Contributing**

We welcome contributions!

**Acknowledgements**

This application uses the following libraries:

- Streamlit
- LangChain
- HuggingFace
- FAISS
- Cohere

**Contact**

For any questions or suggestions, please contact sneh.vora126@gmail.com.

```css
This `README.md` file provides a clear and detailed explanation of the project's purpose, installation steps, usage instructions, and code functionality. It also includes information on contributing and acknowledgments.
```
