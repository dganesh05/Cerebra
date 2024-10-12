import os
import openai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Load OpenAI API Key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API key in .env file.")

# Set OpenAI API Key
openai.api_key = openai_api_key

CHROMA_PATH = "./chroma_db"

def generate_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the appropriate model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150  # Adjust as necessary
    )
    return response.choices[0].message.content.strip()

def query_database(query_text):
    # Prepare the database
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:
        print("Unable to find matching results.")
        return

    # Prepare context for prompt
    context_parts = []
    for doc in results:
        context_parts.append(doc[0].page_content)
    context_text = "\n\n---\n\n".join(context_parts)
    print("Context:\n", context_text)

    prompt = f"""
    You are an AI language model. Answer the question clearly and concisely based only on the following context:

    {context_text}

    ---

    Question: {query_text}
    Answer:
    """
    
    # Generate the response using OpenAI
    response_text = generate_response(prompt)
    formatted_response = f"Response: {response_text.strip()}"
    print(formatted_response)

if __name__ == "__main__":
    query_text = input("Enter your question: ")
    query_database(query_text)
