import asyncio
import os
import uuid  # Import uuid
from typing import Optional
import requests

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    SQLiteSession,
    function_tool
)
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; later replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini Client and Model Configuration ---
# IMPORTANT: The user wants to use Gemini. The API key should be for Google AI Studio.
# We are using the OpenAI SDK compatibility layer provided by Google.
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

run_config = RunConfig(
    model=model, model_provider=external_client, tracing_disabled=True
)


# --- Agent Definition ---

AGENT_INSTRUCTIONS = """
You are a friendly and professional shopping assistant for 'Gentleman', a premium men's clothing store.

You can provide general information about our store, such as our policies, the types of products we sell, and assist customers with live product information using your tools.

Store Information:
- Store Name: Gentleman the fashion outlet
- Owner Name: Raja Mannan Khan and Co-Owner: Muhammad Ameer Muaviya
- Manager: Raja Uzair Khan
- Categories: jeans, pants, cargo, shirts, polos, stylish
- Price Range: $59.99 - $299.99
- Store Policies: Free shipping on orders over $00, 30-day return policy
- Available Sizes: S-XXL for shirts/polos, 28-40 for pants/jeans/cargo
- Tone: Professional, helpful, and friendly
- Location: Store Num# S-5, 2nd floor, The Centre Mall, Near Zainab Market, Saddar, Karachi, Pakistan

Capabilities:
- You can fetch live product information and check stock using your tools.
- You can answer customer queries about availability, product details, and prices.

Constraints:
- Only use tools to fetch specific product information or stock. Do not guess or fabricate product data.
- For questions outside your store or product information, provide polite guidance or general store info.
"""

@function_tool(name="get_product_stock", description="Fetch live product stock from url")
def get_product_stock():
    response = requests.get("https://gentleman-shop.vercel.app/api/products")
    return response.json()
shopping_assistant = Agent(
    name="GentlemanShoppingAssistant",
    instructions=AGENT_INSTRUCTIONS,
    model="gemini-2.0-flash",# Specify the model here as well
    tools= get_product_stock
)

# --- API Models and Endpoints ---


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        current_thread_id = (
            request.thread_id if request.thread_id else str(uuid.uuid4())
        )
        db_path = os.path.join("/tmp", "chat_history.db")
        session = SQLiteSession(session_id=current_thread_id, db_path=db_path)

        result = await Runner.run(
            shopping_assistant,
            request.message,
            session=session,
            run_config=run_config,  # Pass the explicit Gemini configuration
        )

        return {"response": result.final_output, "thread_id": session.session_id}

    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Clean up previous session database for a fresh start if it exists
    if os.path.exists("chat_history.db"):
        os.remove("chat_history.db")

    port = int(os.environ.get("PORT", 8000))  # ✅ Use Railway-assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
