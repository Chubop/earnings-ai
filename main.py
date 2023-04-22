from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import os



def init(text: str):
    loader = TextLoader(text)
    index = VectorstoreIndexCreator().from_loaders([loader])
    while True:
        query = input("Please enter your question regarding the earnings call.\n: ")
        answer = index.query_with_sources(query)
        print(answer['answer'])


if __name__ == '__main__':
    # init('transcripts/lpsn_transcript')
    with open('transcripts/lpsn_transcript', 'r') as outfile:
        for x in outfile.readlines():
            if x != '\n':
                print(x, end='')
