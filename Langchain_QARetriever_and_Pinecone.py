import os
# import getpass
os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_ENV"] = "gcp-starter"

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
import pinecone

loader = TextLoader("./example.txt")
documents = loader.load()
#splitting the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

llm= OpenAI()
embeddings = OpenAIEmbeddings()

pinecone.init(environment="gcp-starter")
index_name = "example-index"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
else:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

chain = load_qa_chain(llm, chain_type="stuff")

class question_answer(BaseModel):
  question: str = Field(..., description = "Question framed.")
  answer: str = Field(..., description = "Answer to the question.")

class output(BaseModel):
  output: list[question_answer] = []

parser = PydanticOutputParser(pydantic_object=output)

prompt = '''You are a dataset creation machine. You make dataset from a given data. Create as much question answer set as you can. Make sure you donot repeat questions and you cover every relevant topic to make the dataset.

Data Provided : {text}

{format_instructions}

Output:'''

dataset = []
chat_llm= ChatOpenAI()
for text in texts:
  _prompt = PromptTemplate(template =  prompt, input_variables = ["text"], partial_variables={"format_instructions": parser.get_format_instructions()})

  _input = _prompt.format_prompt(text = text)
  message = [
    SystemMessage(content = _input.to_string())
  ]

  result = chat_llm(message).content

  parsed_output = parser.parse(result)
  dataset.extend(parsed_output.output)
  print(dataset)

  #QAbot
  question = input("Question > ")
  docs = docsearch.similarity_search(question)
  chain.run(input_documents=docs, question=question)