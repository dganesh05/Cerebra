from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import spacy
from spacy.matcher import Matcher
import re

load_dotenv()

# Load OpenAI API Key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API key in .env file.")

# Set OpenAI API Key
openai.api_key = openai_api_key

DATA_PATH = "C:\\Users\\bajaj_2zp1lvw\\Documents\\Agam Cerebra\\TEMP ONLY!!!!!! Copy of Hackathon Planning 2024.pdf"
CHROMA_PATH = "./chroma_db"

def main():
    generate_data_store()

def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents


def custom_tokenizer(nlp):
    # Compiling expressions from () - to whitespace
    infix_re = re.compile(r'[-()\s]+') 
    # Compiling a tokenizer using the regular expressions listed above
    nlp.tokenizer.infix_finditer = infix_re.finditer
    # Now we can make a phone number - which normally is multiple segments - into one token
    return nlp

def remove_pii(documents):
    nlp = spacy.load("en_core_web_sm")

     # Define regex patterns for phone numbers
    phone_pattern = re.compile(r'''
    \d{3}[-]?\d{4}[-]?\d{3}  # Matches 123-4567-890
    |   # OR
    \(?\d{3}\)?[-.\s]?        # Matches (123)-456-7890 or 123-456-7890
    \d{3}[-.\s]?              # Matches first 3 digits
    \d{4}                     # Matches last 4 digits
''', re.VERBOSE)
    
   # Define regex pattern for email addresses
    email_pattern = re.compile(r'''
        [a-zA-Z0-9._%+-]+         # Username
        @                         # @ symbol
        [a-zA-Z0-9.-]+           # Domain name
        \.[a-zA-Z]{2,}            # Top-level domain
    ''', re.VERBOSE)

    
    for doc in documents:
        text = doc.page_content
        
        # Redact named entities first
        doc_spacy = nlp(text)
        for entity in doc_spacy.ents:
            if entity.label_ in ["PERSON", "MONEY"]:
                text = text.replace(entity.text, "[REDACTED]")

        # Match and replace phone numbers
        text = phone_pattern.sub("[REDACTED]", text)

        # Match and replace email addresses
        text = email_pattern.sub("[REDACTED]", text)

        # Update the document's content
        doc.page_content = text
    return documents


    

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[34]
    print(document.page_content.replace('\n',' '))
    print(document.metadata)

    return chunks

def generate_data_store():
    documents = load_documents()
    documents = remove_pii(documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()