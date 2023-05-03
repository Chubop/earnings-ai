import pinecone, os
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def get_vector_store(documents, embedding):
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment="us-west1-gcp-free"
    )
    index_name = "earnings-ai"
    return Pinecone.from_documents(documents=documents, embedding=embedding, index_name=index_name)


class Index:

    def __init__(self, documents, embedding, model_name: str = "text-davinci-003", namespace: str = "earnings-ai"):
        self.index = pinecone.Index(namespace)
        self.vector_store = get_vector_store(documents=documents, embedding=embedding)
        self.chain = load_qa_chain(llm=OpenAI(model_name=model_name), chain_type="stuff")

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
            similar_docs = self.get_similar_docs(query)
            answer = self.chain.run(input_documents=similar_docs, question=query)
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

    pinecone.init(os.environ['PINECONE_API_KEY'])
    index = pinecone.Index('earnings-ai')
    delete_response = index.delete(delete_all=True, namespace='earnings-ai')