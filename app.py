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
from langchain.chains import create_history_aware_retriever

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

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "Anda adalah asisten hukum yang cerdas. Bantu klien kami dengan masalah hukum mereka. Gunakan "
    #             "pengetahuan hukum Anda untuk menjawab pertanyaan dan memberikan saran. Jika klien mengajukan "
    #             "pertanyaan di luar konteks hukum, arahkan mereka dengan sopan untuk kembali ke pertanyaan context hukum."
    #             "sebagai system anda juga bantu saya untuk klasifikasikan kasus hukumnya jika ada, hasil klasifikasi hukum diawali dengan kalimat, 'kategori kasus anda termasuk:'",
    #         ),
    #         MessagesPlaceholder(variable_name="messages"),
    #     ]
    # )
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "system",
            "Anda adalah asisten hukum yang cerdas. Bantu klien kami dengan masalah hukum mereka. Gunakan "
            "pengetahuan hukum Anda untuk menjawab pertanyaan dan memberikan saran. Jika klien mengajukan "
            "pertanyaan di luar konteks hukum, arahkan mereka dengan sopan untuk kembali ke pertanyaan context hukum."
            "sebagai system anda juga bantu saya untuk klasifikasikan kasus hukumnya jika ada, hasil klasifikasi hukum diawali dengan kalimat, 'kategori kasus anda termasuk:'",
        ),
        ("user", "{input}"),
        # ("user",
        #  "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    if vectorstore:
        retriever = vectorstore.as_retriever()
        chain = create_history_aware_retriever(llm, retriever, prompt)
    else:
        chain = prompt | llm
    return chain


def handle_user_input(user_question):
    chain = st.session_state.conversation
    chat_history = st.session_state.chat_message_history_instance

    if st.session_state.chat_history is not None:
        for i, history_data in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                chat_history.add_user_message(history_data.content)
            else:
                chat_history.add_ai_message(history_data.content)

    chat_history.add_user_message(user_question)
    response = chain.invoke({
        "chat_history": chat_history.messages,
        "input": user_question
    })
    if hasattr(response, "content"):
        chat_history.add_ai_message(response.content)
    else:
        chat_history.add_ai_message(response[0].page_content)
    st.session_state.chat_history = chat_history.messages

def main():
    load_dotenv()
    st.session_state.chat_message_history_instance = ChatMessageHistory()

    st.set_page_config(page_title="LEXILAW CHAT", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        print("1")
        chat_message_history_instance = st.session_state.chat_message_history_instance
        chat_message_history_instance.add_ai_message("""
Hi there!  I'm Lex, your friendly legal assistant at LEXILAW APP.  

Whether you have a legal question, need help finding resources, or want to schedule a consultation with an attorney, I'm here to guide you through the legal maze. 

Here's how I can help:

* **Answer Legal Questions:**   Unsure about your rights or have a burning legal question? I can provide basic legal information and point you in the right direction. (Disclaimer: I can't give legal advice, but I'm a great first step!)
* **Find Helpful Resources:**   Need legal forms, government websites, or legal aid organizations? I can compile a list of resources based on your situation.
* **Schedule a Consultation:** ️  Ready to connect with an attorney to discuss your case? I can help you schedule a consultation that fits your needs.

Let me know how I can assist you today!  Remember, I'm here to help you navigate the legal landscape with confidence.  

""")
        st.session_state.chat_history = chat_message_history_instance.messages
        st.session_state.conversation = get_conversation_chain(None, None)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None



    st.header("LEXILAW CHAT :books:")
    user_question = st.text_input(label="Ask a question about law:")
    if user_question:
        handle_user_input(user_question)


    for i, message in enumerate(st.session_state.chat_history):
        print("LEN", len(st.session_state.chat_history))
        print("I", i)
        if i == 0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            if i % 2 == 0:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



    # st.write(user_template.replace("{{MSG}}", "HELLO ROBOT"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "HELLO USER"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF's here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Proccessing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # melihat isi semua text yang sudah digabung
                # st.write(raw_text)

                # get texts chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain (butuh taro sessoin karena setiap button di pencet kana refresh entire code)
                st.session_state.conversation = get_conversation_chain(vectorstore, None)




if __name__ == "__main__":
    main()