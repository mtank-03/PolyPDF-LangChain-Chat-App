import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import os


def get_pdf_content(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_raw_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
        )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

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

def main():
    load_dotenv()
    st.set_page_config(page_icon=":books:", page_title="Chat With PDF")
    st.write(css, unsafe_allow_html=True)
    user_flag = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        user_flag = True
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiplePDFs :books:")

    with st.sidebar:
        st.title("🤗💬 LLM PDF Chat App")
        add_vertical_space(2)
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process Document'",
            accept_multiple_files=True,
        )


        if st.button("Process Document"):
            with st.spinner("Processing"):
                # get pdf
                raw_text = get_pdf_content(pdf_docs)
                # get text chucks
                text_chunks = get_raw_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        st.markdown(
            """
                        ## About
                        This application is a chatbot powered by the GPT-3.5 language model, built using:
                        - [Streamlit](https://streamlit.io/)
                        - [LangChain](https://python.langchain.com/)
                        - [OpenAI](https://platform.openai.com/docs/models) LLM model
                        - [Hugging Face](https://huggingface.co/docs/huggingface_hub/index) LLM model
                        """
        )

        add_vertical_space(3)
        st.write(
            "Made with ❤️ by [Arjun Haldankar](https://www.linkedin.com/in/arjun-haldankar-8401b9232/)"
        )
        
    
    user_question = st.text_input("Ask a question about your documents:", disabled=user_flag)
    if user_question:
        handle_userinput(user_question)


if __name__ == "__main__":
    main()
