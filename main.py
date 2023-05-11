import vector_db
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from pprint import pprint
from langchain.memory import ConversationBufferMemory


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs


def load_embeddings(model_name: str = "ada"):
    return OpenAIEmbeddings()


def init(directory: str = "transcripts/LPSN/2022", chat_history: list = []):
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
    pprint(Index.ask_question("Could you generate a summary of the call?", verbose=True))
