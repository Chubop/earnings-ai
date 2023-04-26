from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os


def init(text: str, llm):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.create_documents([text])
    loader = TextLoader(text)
    index = VectorstoreIndexCreator().from_loaders([loader])
    while True:
        with get_openai_callback() as cb:
            query = input("Please enter your question regarding the earnings call.\n: ")
            answer = index.query_with_sources(query, llm=llm)
            print(answer['answer'])
            print(cb)


if __name__ == '__main__':
    llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, temperature=0)
    init('transcripts/lpsn_transcript', llm=llm)
    with open('transcripts/lpsn_transcript', 'r') as outfile:
        for x in outfile.readlines():
            if x != '\n':
                print(x, end='')

