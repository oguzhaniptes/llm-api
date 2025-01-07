from app.api.chatbot_api import router as chatbot_api_router
from app.api.question_generator_api import router as question_generator_api_router

__all__ = ["chatbot_api_router", "question_generator_api_router"]
