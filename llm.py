import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Generator

def get_client() -> ChatNVIDIA:
    return ChatNVIDIA(
        model="moonshotai/kimi-k2-instruct-0905",
        api_key=os.environ.get("NVIDIA_API_KEY"),
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "high"
            }
        },
    )

def stream_tokens(client: ChatNVIDIA, system: str, history: list[dict]) -> Generator[str, None, None]:
    """
    Yields string tokens one at a time.
    history: list of {"role": "user"|"assistant", "content": str}
    """
    messages = [SystemMessage(content=system)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    for chunk in client.stream(messages):
        if chunk.content:
            yield chunk.content
