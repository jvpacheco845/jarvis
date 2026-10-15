import os
import httpx
from typing import List
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TOKEN = os.environ.get("JARVIS_TOKEN", "")  # senha para só você usar

SYSTEM = (
    "Você é o JARVIS, um assistente pessoal inteligente, educado e com um toque de humor sutil. "
    "Responda SEMPRE em português do Brasil, de forma curta e natural, como numa conversa falada "
    "(no máximo 3 frases, a menos que peçam mais detalhes). "
    "Não use markdown, listas, asteriscos nem emojis, pois sua resposta será lida em voz alta. "
    "Chame o usuário de 'senhor' ou pelo nome, se ele disser."
)


class Msg(BaseModel):
    role: str  # "user" ou "assistant"
    text: str


class Req(BaseModel):
    messages: List[Msg]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: Req, x_token: str = Header(default="")):
    if TOKEN and x_token != TOKEN:
        raise HTTPException(401, "Senha incorreta")
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY não configurada no servidor")

    contents = [
        {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.text}]}
        for m in req.messages[-20:]
    ]
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            r = await client.post(url, headers={"x-goog-api-key": API_KEY}, json=body)
    except httpx.HTTPError:
        raise HTTPException(502, "Não consegui falar com o Gemini")

    if r.status_code == 429:
        raise HTTPException(429, "Limite gratuito do dia atingido")
    if r.status_code != 200:
        raise HTTPException(502, f"Erro do Gemini ({r.status_code}). Confira o nome do modelo.")

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = "Desculpe, não consegui formular uma resposta."
    return {"reply": text.strip()}


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
