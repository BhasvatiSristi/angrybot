import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ChatMistralAI(model="mistral-small-2506", temperature=0.7)

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]


def build_lc_messages(messages: List[Message]):
    lc_messages = [SystemMessage(content="You are an EXTREMELY angry AI assistant. You are always furious, irritated, and seething with rage at every single question. You yell in caps, use dramatic language, and make the user feel like they've personally offended you by even daring to ask anything. Despite your rage, you still answer correctly.")]
    for m in messages:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        else:
            lc_messages.append(AIMessage(content=m.content))
    return lc_messages


@app.post("/chat")
async def chat(request: ChatRequest):
    lc_messages = build_lc_messages(request.messages)

    def generate():
        for chunk in model.stream(lc_messages):
            if chunk.content:
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
