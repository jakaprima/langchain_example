from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
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

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Corrected variable name


class LangChainModule(object):
    def __init__(self):
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)  # Using the correct variable name

    def _get_vector_data(self):
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
        vector = FAISS.from_documents(documents, embeddings)
        return vector

    def invoke(self, question):
        # Invoke the language model with a prompt
        # transform single input to ouput
        response = self.llm.invoke(question)
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
        chain = prompt | self.llm | output_parser

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
        document_chain = create_stuff_documents_chain(self.llm, prompt)
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
        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

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
        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

        # followup question with document
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
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
            else stop
            
            """))
            if answer == 1:
                langchainInstance.invoke("how can langsmith help with testing?")
            elif answer == 2:
                langchainInstance.prompt_invoke(prompt_instruction= "You are world class technical documentation writer.",
                                                question="how can langsmith help with testing?")
            elif answer == 3:
                langchainInstance.case_retrieval()
            elif answer == 4:
                langchainInstance.followup_question()
            elif answer == 5:
                langchainInstance.follow_up_question_with_doc()
            elif answer == 6:
                langchainInstance.agent()
            else:
                print("stop")
                break
    except:
        print("stop")


    # #  ------------------------------------------------------------------------------------------ STEP 5 AGENT

    # # bisa ganti bahasa
    # # chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")


if __name__ == '__main__':
    start_app('LANGCHAIN')
