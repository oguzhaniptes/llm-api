from fastapi import FastAPI
from app.api import chatbot_api, question_generator_api


def create_app() -> FastAPI:
    app = FastAPI(title="FastAPI Projesi")

    # API modüllerini ekle
    app.include_router(
        question_generator_api.router, prefix="/question", tags=["API 1"]
    )
    app.include_router(chatbot_api.router, prefix="/chat", tags=["API 2"])

    @app.get("/")
    async def root():
        return {"message": "Hoş geldiniz! FastAPI çalışıyor."}

    return app
