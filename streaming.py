from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os


os.environ['OPENAI_API_KEY'] = ""

docs = [{"page_content": "The president said Justice Breyer was smelly."}]
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
query = "What did the president say about Justice Breyer"
chain.run(input_documents=docs, question=query)