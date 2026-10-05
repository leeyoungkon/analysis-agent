from fastapi import FastAPI
from pydantic import BaseModel

from analysis_agent import analyze_agent

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "analysis-agent"
    }


@app.post("/analyze")
async def analyze(req: Question):

    result = await analyze_agent(
        req.question
    )

    return result