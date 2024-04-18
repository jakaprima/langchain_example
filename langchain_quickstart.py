from dotenv import load_dotenv
import os
import bs4

from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeVectorStore

from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import traceback

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Corrected variable name
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


class LangChainModule(object):
    def __init__(self):
        # temperature is for setting ai creative generative response
        # llm_chat temperature 0.7
        # llm_general more flexible can more customable for text generation, translation, and question answer
        self.llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)  # Using the correct variable name
        self.llm_general = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)

    def _get_vector_data(self, docs=None, question=None):
        if docs is None:
            # load data yang akan di index dengan beautifulsoup
            loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
            docs = loader.load()

            # selanjutnya butuh index ke vectorstore, component yang dibutuhkan
            # embedding model & vectorstore
            # gunakan class ini untuk embedding(sematkan pelengkap) model untuk memberikan document ke vectorstore
            embeddings = OpenAIEmbeddings()

            # gunakan simple local vectorstore, FAISS
            # Facebook AI Similarity Search (Faiss)
            # build our index
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(docs)
            faiss_index = FAISS.from_documents(documents, embeddings)
        else:
            faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings())
            docs = faiss_index.similarity_search(question, k=2)
            # for doc in docs:
            #     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
        return faiss_index

    def _get_vector_data_uud_by_faiss(self, texts):
        embeddings = OpenAIEmbeddings()
        vector = FAISS.from_documents(texts, embeddings)
        return vector

    def _get_vector_data_uud_by_pinecone(self, texts, question):
        embeddings = OpenAIEmbeddings()
        pinecone_index = "chatbotjaka"

        pineconeInstance = Pinecone(api_key=PINECONE_API_KEY)
        index = pineconeInstance.Index(pinecone_index)
        space_name = "my_space"

        # Create the Pinecone vector store
        vectorstore = PineconeVectorStore.from_documents(
            texts, embeddings, index_name=pinecone_index, pinecone_api_key=PINECONE_API_KEY
        )
        print("VECTOR STORE", vectorstore)

        # Perform similarity search
        similar_documents = vectorstore.similarity_search(question)
        return vectorstore, similar_documents

    def invoke(self, question):
        # Invoke the language model with a prompt
        # transform single input to ouput
        response = self.llm_chat.invoke(question)
        print("RESPONSE INVOKE:", response.content)
        return response

    def prompt_invoke(self, prompt_instruction, question):
        # Define a chat prompt template
        # """
        # another example:
        # template = ChatPromptTemplate.from_messages([
        #     ("human", "Hello, how are you?"),
        #     ("ai", "I'm doing well, thanks!"),
        #     ("human", "That's good to hear."),
        # ])
        # """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_instruction),
            ("user", "{input}")
        ])

        # Define the conversation chain (mengatur alur percakapan)
        output_parser = StrOutputParser()
        chain = prompt | self.llm_chat | output_parser

        # Invoke the chain with a user input
        chain_response = chain.invoke({"input": question})
        print("Chain response:", chain_response)
        return chain_response

    def case_retrieval(self, question="how can langsmith help with testing?"):
        """
        digunakan untuk menambahkan konteks ke LLM. berguna ketika kamu punya banyak data untuk di berikan ke LLM
        gunakan untuk meretriever untuk mengambil hanya potongan yang paling relevant dan memberikannya pada LLM
        retrieval bisa berupa sql table, dari internet, dll
        :return:
        """

        # create retrieval chain -> take question -> look relevant from documents, dan berikan document dengan original question ke LLM
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")
        # Create a chain for passing a list of Documents to a model.
        document_chain = create_stuff_documents_chain(self.llm_chat, prompt)
        # print("ANSWER 10", document_chain)

        # passing in documents directly
        # answer = document_chain.invoke({
        #     "input": question,
        #     "context": [Document(page_content="langsmith can let you visualize test results")]
        # })
        vector = self._get_vector_data()

        # document first come from the retriever
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        # print("ANSWER 12", retrieval_chain)

        # This answer should be much more accurate!
        response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
        print("ANSWER:", response["answer"])
        return response

    def followup_question(self):
        """
        Conversation Retrieval Chain
        EXAMPLE OF CHAINS - WHERE EACH STEP IS KNOWN AHEAD OF TIME
        """
        # chain yang dibuat diatas semua untuk menjawab 1 pertanyaan.
        # one of the main types of LLM app people build chat bots
        # create followup question
        vector = self._get_vector_data()
        retriever = vector.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(self.llm_chat, retriever, prompt)

        # followup question
        chat_history = [
            HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")
        ]
        response = retriever_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        print("ANSWER OF FOLLOW UP QUESTION:", response)
        return response

    def follow_up_question_with_doc(self):
        vector = self._get_vector_data()
        retriever = vector.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(self.llm_chat, retriever, prompt)

        # followup question with document
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(self.llm_chat, prompt)
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        chat_history = [
            HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!"),

        ]
        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        print("RESPONSE WITH DOC :", response)
        return response

    def agent(self):
        # AGENT
        # where LLM decides what step to take
        # EXAMPLE:
        # berikan agent access ke 2 tools:
        # 1. retriever yang sudah dibuat
        # 2. search tool. agar dengan mudah menjawab pertanyaan yang membutuhkan up-to-date information
        # """
        vector = self._get_vector_data()
        retriever = vector.as_retriever()

        # setup
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )
        retriever = TavilySearchAPIRetriever(k=3)

        response = retriever.invoke("what year was breath of the wild released?")
        print("RESPONSE AGENT", response)

        search = TavilySearchResults()
        tools = [retriever_tool, search]

        # Get the prompt to use - you can modify this!
        prompt = hub.pull("hwchase17/openai-functions-agent")

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        answer_1 = agent_executor.invoke({"input": "how can langsmith help with testing?"})
        answer_2 = agent_executor.invoke({"input": "what is the weather in SF?"})
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
        answer_3 = agent_executor.invoke({
            "chat_history": chat_history,
            "input": "Tell me how"
        })
        print("ANSWER 1", answer_1)
        print("ANSWER 2", answer_2)
        print("ANSWER 3", answer_3)

        # bisa ganti bahasa
        # chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

    def scan_in_pdf_with_pinecone(self):
        # Define paths (replace with your actual file paths)
        default_question = "ada di bab berapa tentang HAL KEUANGAN"
        print(f"QUESTION: (default: {default_question})")
        question = input()
        if question == "":
            question = default_question

        print(os.getcwd())
        pdf_loader = PyPDFLoader(os.getcwd() + "/UUD45_ASLI.pdf")
        data = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

                <context>
                {context}
                </context>

                Question: {input}""")
        document_chain = create_stuff_documents_chain(self.llm_chat, prompt)
        vector, similar_docs = self._get_vector_data_uud_by_pinecone(texts=texts, question=question)
        # direct check similarity not in OPENAI
        document_chain.invoke({
            "input": question,
            "context": similar_docs
        })

        # search in OPENAI by model text-embedding-data
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": question})

        print("ANSWER:", response)
        return response

    def scan_in_pdf_with_faiss(self):
        # Define paths (replace with your actual file paths)
        default_question = "ada di bab berapa tentang HAL KEUANGAN"
        print(f"QUESTION: (default: {default_question})")
        question = input()
        if question == "":
            question = default_question

        print(os.getcwd())
        pdf_loader = PyPDFLoader(os.getcwd() + "/UUD45_ASLI.pdf")
        pages = pdf_loader.load_and_split()
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

                <context>
                {context}
                </context>

                Question: {input}""")
        document_chain = create_stuff_documents_chain(self.llm_chat, prompt)
        index = self._get_vector_data(docs=pages, question=question)
        retriever = index.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # This answer should be much more accurate!
        response = retrieval_chain.invoke({"input": question})
        print("ANSWER:", response)
        return response

    def generate_pet_name(self, animal_type: str, pet_color: str):
        # without prompt_templates
        # question = "I have a dog pet and i want a cool name for it. suggest me five cool names for my pets"
        # names = self.llm_general(question)

        # with prompt_templates
        # reprepose if want to add animal type and color (dynamic text to intructions)
        prompt_template_name = PromptTemplate(
            input_variables=["animal_type", "pet_color"],
            template="I have {animal_type} pet and i want a cool name for it, it is {pet_color} in color. "
                     "suggest me five cool names for my pet",
        )
        name_chain = LLMChain(llm=self.llm_general, prompt=prompt_template_name)
        response = name_chain({"animal_type": animal_type, "pet_color": pet_color})
        print("ANSWER: ", response)
        return response["text"]

    def langchain_agent_pet(self):
        llm = OpenAI(temperature=0.5)
        tools = load_tools(["wikipedia", "llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        result = agent.run("What is the average age of the dog? Multiply the age by 3")
        print("RESULT: ", result)

    def youtube_assistant_db_from_youtube_content(self, video_url: str, question: str):
        # ask question to video
        embeddings = OpenAIEmbeddings()
        loader = YoutubeLoader.from_youtube_url(youtube_url=video_url)
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)
        print("DOCS SPLIT", docs)
        db = FAISS.from_documents(docs, embeddings)
        print("db", db)
        return db

    def youtube_assistant_get_response_from_query(self, db, query, k=4):
        # text-davinci can handle 4097 tokens
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])

        # biar ga di charge pake akun sendiri
        api_key_user = input("YOUR OPENAI API KEY: \n")
        if api_key_user is None:
            api_key_user = OPENAI_API_KEY
        llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=api_key_user)
        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful youtube assistant that can answer questions about videos
            based on the video's transcript.
            
            Answer the following question: {question}
            By searching the following video transcript: {docs}
            Only use the factual information from the transcript to answer the question.
            if you feel like you don't have enought information to answer the question, say " I don't know."
            
            Your answers should be detailed.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        print("ANSWER : ", response)
        return response

    def question_answer_with_rag(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        print(len(docs[0].page_content))
        print(docs[0].page_content[:500])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splits = text_splitter.split_documents(docs)
        print(len(splits))
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        response = rag_chain.invoke("What is Task Decomposition?")
        print("RESPONSE: ", response)

        # cleanup
        # vectorstore.delete_collection()


def start_app(name):
    """
    --------------------------------------------------------------------------------------------------------- LLM CHAIN
    """
    langchainInstance = LangChainModule()
    try:
        while True:
            answer = int(input("""
            MAU TEST NO BERAPA?
            1. basic invoke
            2. basic prompt intruction and question
            3. retrieval
            4. followup question
            5. followup question with docs
            6. agent
            7. question answer cari di uud doc pdf dengan FAISS
            8. question answer cari di uud doc pdf dengan PINECONE
            9. generate me 5 pets names
            10. agent dog age information
            11. youtube assistant
            12. Q&A with RAG
            else stop
            
            """))
            if answer == 1:
                langchainInstance.invoke("how can langsmith help with testing?")
            elif answer == 2:
                langchainInstance.prompt_invoke(
                    prompt_instruction="You are world class technical documentation writer.",
                    question="how can langsmith help with testing?")
            elif answer == 3:
                langchainInstance.case_retrieval()
            elif answer == 4:
                langchainInstance.followup_question()
            elif answer == 5:
                langchainInstance.follow_up_question_with_doc()
            elif answer == 6:
                langchainInstance.agent()
            elif answer == 7:
                langchainInstance.scan_in_pdf_with_faiss()
            elif answer == 8:
                langchainInstance.scan_in_pdf_with_pinecone()
            elif answer == 9:
                animal_type = input("GIVE ME CLUE YOUR ANIMAL TYPE, default: cat\n your answer:")
                if animal_type is None:
                    animal_type = "cat"
                animal_color = input("GIVE ME CLUE YOUR ANIMAL COLOR, default: black\n your answer:")
                if animal_type is None:
                    animal_type = "black"
                langchainInstance.generate_pet_name(animal_type, animal_color)
            elif answer == 10:
                langchainInstance.langchain_agent_pet()
            elif answer == 11:
                print("YOUTUBE ASSISTANT")
                youtube_url = input("GIVE ME YOUR YOUTUBE URLS: \n")
                question = input("ASK ME ABOUT THE VIDEO: \n")
                db = langchainInstance.youtube_assistant_db_from_youtube_content(youtube_url, question)
                answer = langchainInstance.youtube_assistant_get_response_from_query(db, question)
            elif answer == 12:
                print("USE CASE QUESTION & ANSWER with RAG (Retrieval and Generation)")
                langchainInstance.question_answer_with_rag()

            else:
                print("stop")
                break
    except Exception as e:
        print("TRACEBACK", traceback.format_exc())
        print("stop", e)

    # #  ------------------------------------------------------------------------------------------ STEP 5 AGENT

    # # bisa ganti bahasa
    # # chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")


if __name__ == '__main__':
    start_app('LANGCHAIN')
