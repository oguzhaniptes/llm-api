import gc
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(model):
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model


app = FastAPI()

conversation_history = []


class InputData(BaseModel):
    text: str


def format_conversation(history):
    formatted_text = ""
    for msg in history:
        if msg.startswith("User: "):
            formatted_text += msg + "\n"
        else:
            formatted_text += "Assistant: " + msg + "\n"
    return formatted_text.strip()


def chatbot_response(history):
    set_device(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    formatted_history = format_conversation(history)

    if len(formatted_history) > 512:
        formatted_history = formatted_history[-512:]

    inputs = tokenizer(
        [formatted_history], return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    if inputs["input_ids"].size(1) > 512:
        raise ValueError("Input sequence exceeds maximum length of 512 tokens.")

    reply_ids = model.generate(
        **inputs, max_length=512, min_length=10, pad_token_id=tokenizer.eos_token_id
    )

    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    if torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()

    gc.collect()

    return reply.strip()


@app.post("/chat")
def chat(input_data: InputData):
    global conversation_history
    user_message = f"User: {input_data.text}"
    conversation_history.append(user_message)

    response = chatbot_response(conversation_history)
    conversation_history.append(f"Assistant: {response}")

    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    gc.collect()

    return {"response": response, "history": conversation_history}
