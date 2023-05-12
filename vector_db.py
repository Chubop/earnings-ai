import pinecone, os
from langchain import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

from langchain.callbacks import get_openai_callback

template = """
You are a tool designed to analyze financial documents (earnings call transcripts, 10-Qs, etc)
and to answer questions regarding them. Construct your answers in the tone of a financial analyst.
Answer the question based on the context below. If the question cannot be answered using the information
provided, respond with "I don't know."

Context: {context}

Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template=template
)


def get_vector_store(documents, embedding, from_existing_index=True):
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment="us-west1-gcp-free"
    )
    index_name = "earnings-ai"
    if from_existing_index:
        return Pinecone.from_existing_index(embedding=embedding, index_name=index_name)
    return Pinecone.from_documents(documents=documents, embedding=embedding, index_name=index_name)


class Index:

    def __init__(self, documents, embedding, chat_history = [], model_name: str = "gpt-3.5-turbo", namespace: str = "earnings-ai"):
        def get_chat_history(inputs) -> str:
            res = []
            for ai, human in inputs:
                res.append(f"Human:{human}\nAI:{ai}")
            return "\n".join(res)

        self.index = pinecone.Index(namespace)
        self.vector_store = get_vector_store(documents=documents, embedding=embedding)

        llm = OpenAI(
            model_name=model_name,
            temperature=0)
        self.chat_history = chat_history
        self.chain = load_qa_chain(
            llm=llm,
            chain_type="map_reduce",
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            # qa_prompt=prompt_template,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            get_chat_history=get_chat_history
        )

        self.embedding = embedding
        self.documents = documents
        self.model_name = model_name
        self.namespace = namespace

    def get_similar_docs(self, query, k=2, score=False):
        if score:
            similar_docs = self.vector_store.similarity_search_with_score(query=query, k=k)
        else:
            similar_docs = self.vector_store.similarity_search(query=query, k=k)
        return similar_docs

    def ask_question(self, query: str = None, verbose=False):
        with get_openai_callback() as cb:
            answer = self.qa({"question": query, "chat_history": self.chat_history})
            return [answer, cb if verbose else None]

    def query(self):
        query_response = self.index.query(
            namespace=self.namespace,
            top_k=10,
            include_values=True,
            include_metadata=True,
        )
        return query_response


if __name__ == '__main__':
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'])

    index = pinecone.Index('earnings-ai')
    delete_response = index.delete(delete_all=True, namespace='earnings-ai')
