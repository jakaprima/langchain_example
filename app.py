import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ChatMessageHistory


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    # chunk_overlap untuk memastikan dari 200 character sebelumnya, dan melanjutkan dari pemotongan sebelumnya
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    # berbayar via models embedding openai
    embeddings = OpenAIEmbeddings()
    # free: pros lama downloadnya, terlalu lama
    # Downloading torch-2.2.2-cp38-cp38-manylinux1_x86_64.whl (755.5 MB)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, question):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm
    return chain


def handle_user_input(user_question):
    chain = st.session_state.conversation

    chat_history = ChatMessageHistory()
    if st.session_state.chat_history is not None:
        for i, history_data in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                chat_history.add_user_message(history_data.content)
            else:
                chat_history.add_ai_message(history_data.content)

    chat_history.add_user_message(user_question)
    response = chain.invoke({"messages": chat_history.messages})
    chat_history.add_ai_message(response.content)
    st.session_state.chat_history = chat_history.messages

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title="LEXILAW CHAT", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(None, None)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("LEXILAW CHAT :books:")
    user_question = st.text_input(label="Ask a question about law:")
    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}", "HELLO ROBOT"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "HELLO USER"), unsafe_allow_html=True)

    # with st.sidebar:
    #     st.subheader("Your Documents")
    #     pdf_docs = st.file_uploader("Upload your PDF's here and click on 'Process'", accept_multiple_files=True)
    #
    #     if st.button("Process"):
    #         with st.spinner("Proccessing"):
    #             # get pdf text
    #             raw_text = get_pdf_text(pdf_docs)
    #             # melihat isi semua text yang sudah digabung
    #             # st.write(raw_text)
    #
    #             # get texts chunks
    #             text_chunks = get_text_chunks(raw_text)
    #             # st.write(text_chunks)
    #
    #             # create vector store
    #             vectorstore = get_vectorstore(text_chunks)
    #
    #             # create conversation chain (butuh taro sessoin karena setiap button di pencet kana refresh entire code)
    #             st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__ == "__main__":
    main()