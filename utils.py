import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


def get_text(pdf_doc):
    '''
    A function to get the text from uploaded pdf document.

    Returns a single string of extracted texts.
    '''
    text = ""
    for doc in pdf_doc:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    '''
    Takes the raw text and splits the large string into smaller chunks (1000 chars in this case)

    Returns text chunks
    '''
    text_splitter=CharacterTextSplitter(separator='\n', 
                                   chunk_size = 1000,
                                    chunk_overlap = 200,
                                    length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    '''
    Takes in text chunks, embedds them and stores them  in a vector store. Uses HugginFace's "BAAI/bge-base-en" model. 

    Returns vector store 
    '''
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model_norm = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
    )

    embeddings = model_norm
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_converse_chain(vectorstore):
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key = 'chat_history', return_message = True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                                memory=memory,
                                                                retriever=vectorstore.as_retriever())
        return conversation_chain







