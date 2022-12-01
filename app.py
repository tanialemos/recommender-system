from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Movie(BaseModel):
    title: str

@app.get("/")
async def root():
    return {"Movie recommender alive and kicking!"}

        

