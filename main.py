import os
import chainlit as cl
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GEMINI API Key from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Step 2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# Step 3: Config
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Step 4: Agent — No tools
agent = Agent(
    name="Smart Assistant",
    instructions=(
        "You are a helpful, friendly assistant. Answer questions clearly and provide support on any topic."
    ),
    model=model,
)

# Step 5: On Chat Start
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "Hi, I'm your AI assistant 🤖!\n\n"
            "Ask me anything — I'm here to help with general knowledge, explanations, tasks, and more!"
        )
    ).send()

# Step 6: On Message
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})

    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)