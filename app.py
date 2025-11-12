from flask import Flask, render_template, request
from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from prompts import qa_system_prompt, contextualize_q_system_prompt


app = Flask(__name__)

# set environment keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

FAISS_PATH = "faiss"    # folder where faiss.index, vectors.npy, metas.json are stored

# store chat history (in memory only)
conversation_store = {}
chat_history = []

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in conversation_store:
        conversation_store[session_id] = ChatMessageHistory()
    return conversation_store[session_id]


# ✅ Load FAISS index from your existing files
def load_faiss():
    return FAISS.load_local(
        FAISS_PATH,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )


def get_retriever():
    db = load_faiss()
    return db.as_retriever()


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")

    question = request.form["question"]

    retriever = get_retriever()

    # Contextual question rewriting
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Final answer generation
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, answer_chain)

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": "user1"}}
    )

    chat_history.append(question)
    chat_history.append(response["answer"])

    return render_template("chat.html", chat_history=chat_history)


if __name__ == "__main__":
    app.run(debug=True)
