from fastapi import FastAPI
from pydantic import BaseModel

class ChromaDB:
    def __init__(self):
        pass

    def get_context(self, question, k =5):
        pass


# Initialization
app = FastAPI()
db = ChromaDB()



class GenerationRequest(BaseModel):
    question: str

@app.post("/generation")
async def generate(request: GenerationRequest):
    question = request.question

    context = db.get_context(question,k=5)

    context = []
    response = ""

    return {"context": context, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
