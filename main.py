import openai, vector_db
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs


def load_embeddings(model_name: str = "ada"):
    return OpenAIEmbeddings()


def init(directory: str = "transcripts/LPSN/2022"):
    documents = load_docs(directory=directory)
    docs = split_docs(documents)
    embedding = load_embeddings()
    Index = vector_db.Index(documents=docs, embedding=embedding)
    return Index


if __name__ == '__main__':
    # llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, temperature=0)
    Index = init(directory="transcripts/AAPL/2023")
    # query = Index.index.query()
    # print(query)
    print(Index.ask_question("Who is John Collins?", verbose=True))

