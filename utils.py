from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
import os
import openai
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import streamlit as st
os.environ["OPENAI_API_KEY"]  = ""
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = 'db'

# embedding = OpenAIEmbeddings()

# Print number of txt files in directory
loader = DirectoryLoader('docs', glob="./*.txt")
doc = loader.load( )
# len(doc)

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(doc)

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0, 
                                                  model_name='gpt-3.5-turbo-1106'),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def find_match(inputs):
    # input_em = model.encode(input).tolist()
    # result = index.query(input_em, top_k=2, includeMetadata=True)
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    llm_response = qa_chain(inputs)
    return llm_response

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string