import os
import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

# Инициализация FastAPI
app = FastAPI(title="Halyk Voice Router API")

# Настройка асинхронного клиента
# API_KEY лучше задавать через терминал: set API_KEY=твой_ключ (Windows) или export API_KEY=твой_ключ (Mac/Linux)
client = AsyncOpenAI(
    api_key=os.getenv("API_KEY", "your_api_key_here"),
    base_url="https://api.x.ai/v1"  # Эндпоинт для Grok. Можно заменить на Groq или стандартный OpenAI
)

# Pydantic модели для строгой валидации входящих и исходящих данных
class VoiceRequest(BaseModel):
    text: str

class RouterResponse(BaseModel):
    intent: str
    latency_ms: float = 0.0

# Жесткий системный промпт для удержания модели в рамках JSON-формата
SYSTEM_PROMPT = """
Ты — умный маршрутизатор голосового робота Halyk Bank.
Анализируй текст клиента (на казахском, русском или смешанном языках) и возвращай ТОЛЬКО JSON с ключом "intent".

Доступные интенты:
- "card_block": потеря/кража карты, мошенники, блокировка.
- "balance_check": проверка счета, баланс, зарплата, остаток.
- "unknown": всё остальное, сложные вопросы.

Формат ответа: {"intent": "название_интента"}
"""

async def fetch_intent_from_llm(text: str) -> str:
    """Асинхронный вызов LLM с принудительным JSON форматом."""
    response = await client.chat.completions.create(
        model="grok-beta",  # Замените на нужную модель, например llama3-8b-8192, если используете Groq
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,  # Нулевая температура для предсказуемости
        max_tokens=15     # Ограничение длины ответа для скорости
    )
    
    result_text = response.choices[0].message.content
    try:
        data = json.loads(result_text)
        return data.get("intent", "unknown")
    except json.JSONDecodeError:
        return "unknown"

@app.post("/route", response_model=RouterResponse)
async def route_voice_query(request: VoiceRequest):
    """
    Эндпоинт маршрутизации с защитой от долгих ответов LLM.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Жесткий таймаут: если модель не отвечает за 800 мс, запрос прерывается
        intent = await asyncio.wait_for(fetch_intent_from_llm(request.text), timeout=0.8)
    except asyncio.TimeoutError:
        intent = "unknown"
    except Exception as e:
        print(f"Ошибка вызова LLM API: {e}")
        intent = "unknown"
        
    end_time = asyncio.get_event_loop().time()
    latency = round((end_time - start_time) * 1000, 2)
    
    return RouterResponse(intent=intent, latency_ms=latency)