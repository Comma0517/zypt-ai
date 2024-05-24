import streamlit as st
import os
import tempfile
import pinecone

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

pinecone.init(api_key=st.secrets['PINECONE_KEY'], environment=st.secrets['PINECONE_ENV'])

os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets['AZURE_OPENAI_ENDPOINT']
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets['AZURE_OPENAI_API_KEY']

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into Cosmos DB
def vectorize_text(uploaded_file, vector_store):
    if uploaded_file is not None:
        
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Load the PDF
        docs = []
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 3500,
            chunk_overlap  = 200
        )

        aoai_embeddings = AzureOpenAIEmbeddings(
            azure_deployment="embedding",
            openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
            chunk_size=10
        )

        # Vectorize the PDF and load it into the Cosmos DB Vector Store
        pages = text_splitter.split_documents(docs)
        Pinecone.from_documents(documents=pages, embedding=aoai_embeddings, index_name=st.secrets['PINECONE_INDEX'], namespace=st.secrets['PINECONE_NAMESPACE'])
        st.info(f"{len(pages)} pages loaded.")

# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """You're a helpful AI assistent tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("system", template)])
prompt = load_prompt()

# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return AzureChatOpenAI(
        openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
        azure_deployment="gpt",
        temperature=0,
    )
chat_model = load_chat_model()

# Cache the Cosmos DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Vector DB')
def load_vector_store():
    # Connect to the Vector Store
    aoai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embedding",
        openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
        chunk_size=10
    )
    vector_store = Pinecone.from_existing_index(
			index_name=st.secrets['PINECONE_INDEX'],
			embedding=aoai_embeddings,
			namespace=st.secrets['PINECONE_NAMESPACE'],
		)
    return vector_store
vector_store = load_vector_store()

# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever():
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever
retriever = load_retriever()

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader('Upload a document for additional context', type=['pdf'])
        submitted = st.form_submit_button('Save to Cosmos DB')
        if submitted:
            vectorize_text(uploaded_file, vector_store)

    # st.write("Integrate with your cloud environment")


# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if question := st.chat_input("Send Message here!"):
    
    # Store the user's question in a session object for redrawing next time
    st.session_state.messages.append({"role": "human", "content": question})

    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate the answer by calling OpenAI's Chat Model
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['question']),
        'question': lambda x: x['question']
    })
    chain = inputs | prompt | chat_model
    response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
    answer = response.content

    # Store the bot's answer in a session object for redrawing next time
    st.session_state.messages.append({"role": "ai", "content": answer})

    # Write the final answer without the cursor
    response_placeholder.markdown(answer)