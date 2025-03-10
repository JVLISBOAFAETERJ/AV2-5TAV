import streamlit as st

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from htmlTemplates import user_template, bot_template, css

import logging

logging.basicConfig(level="DEBUG")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(text_chunks):
    embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(
        api_version="2023-12-01-preview", azure_deployment="gpt-4-turbo-2024-04-09"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Assistente Jurídico", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Assistente Jurídico")
    user_question = st.text_input("Faça uma pergunta sobre seus documentos:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Seus documentos")
        pdf_docs = st.file_uploader("Envie seus documentos", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processando"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
