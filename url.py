import streamlit as st
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader #, SeleniumURLLoader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template, css
from dotenv import load_dotenv

def load_urls(urls):
    loaders=UnstructuredURLLoader(urls=urls)
    return loaders     


def split_text(loaders):
    
    data=loaders.load()

    text_spplit=CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=200
    )
    
    docs = text_spplit.split_documents(data)
    
    return docs


def embed_in_DB(docs):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs,embeddings)
    return vectorstore


def chain(vectorstore):
    llm=ChatOpenAI(temperature=0,model='gpt-3.5-turbo')

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def save_urls_to_file(url_list):
    # Save the URLs to a text file called "urls.txt"
    with open("urls.txt", "w") as f:
        for url in url_list:
            f.write(f"{url}\n")

def load_urls_from_file():
    # Load the URLs from the "urls.txt" file
    try:
        with open("urls.txt", "r") as f:
            url_list = [url.strip() for url in f.readlines()]
    except FileNotFoundError:
        url_list = []
    return url_list

def main():
    load_dotenv()
    st.set_page_config(page_title="QnA ChatBot",
                       page_icon=":book:")
    st.write(css,unsafe_allow_html=True)

    with st.sidebar:
        st.title("Enter URLs separated by commas:")

        # Load the URLs from the file on app startup
        url_list = load_urls_from_file()

        # Create a text area for entering the URLs separated by commas
        url_input = st.text_area("")

        # Create a horizontal layout for the buttons
        col1, col2 = st.columns(2)

        # Process button in the first column
        if col1.button("Add URL"):
            if url_input:
                # Split the input string by commas and remove leading/trailing whitespaces
                urls = [url.strip() for url in url_input.split(",")]

                # Append each URL to the list
                url_list.extend(urls)

                # Save the updated list to the file
                save_urls_to_file(url_list)

                # Display success message
                st.success(f"{len(urls)} URLs have been added to the list.")

        # Clear List button in the second column
        if col2.button("Clear List"):
            # Clear the URL list and the file
            url_list.clear()
            save_urls_to_file(url_list)
            st.warning("URL list has been cleared.")

        # Display the current list of URLs
        st.header("List of URLs:")
        for index, url in enumerate(url_list):
            st.write(f"{index + 1}. {url}")

    st.header("QnA ChatBot:")
    user_question = st.text_input("Enter your query:")
    col1, col2 = st.columns(2)
    if col1.button("Ask"):
        if user_question:
            handle_userinput(user_question)
            
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # urls=[
    # 'https://geu.ac.in/scholarships/',
    # 'https://geu.ac.in/about-us/',
    # 'https://geu.ac.in/about-us/presidents-message/',
    # ]
    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader(
    #         "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if col2.button("Embed"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = load_urls(url_list)

            # get the text chunks
            text_chunks = split_text(raw_text)

            # create vector store
            vectorstore = embed_in_DB(text_chunks)

            # create conversation chain
            st.session_state.conversation = chain(vectorstore)


if __name__ == "__main__":
    main()
