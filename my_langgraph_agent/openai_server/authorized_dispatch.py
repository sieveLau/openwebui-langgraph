from .auth import validate_api_key
from fastapi import APIRouter, Depends
from .schema import ChatCompletionRequest
from my_langgraph_agent.langgraph_agent import GraphState
from my_langgraph_agent import app
from langchain_core.messages import AIMessage
from fastapi.responses import StreamingResponse
import json, time

router = APIRouter(dependencies=[Depends(validate_api_key)])


def dict_to_sse(d: dict):
    return "data: " + json.dumps(d) + "\n\n"


@router.post("/v1/chat/completions")
async def my_endpoint(payload: ChatCompletionRequest):
    created = time.time()
    messages = [[msg.role, msg.content] for msg in payload.messages]
    state: GraphState = {
        "messages": messages,
        "messages_with_think": messages,
        "id": "",
        "model": "",
        "created": int(created),
        "finish_reason": "",
    }
    if payload.stream:

        async def event_stream():
            first_token = True
            id = None
            model_name = None
            async for msg, metadata in app.astream(input=state, stream_mode="messages"):
                if msg.content:  # type: ignore
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if first_token:
                            yield dict_to_sse(
                                {
                                    "id": msg.id,
                                    "created": created,
                                    "object": "chat.completion.chunk",
                                    "model": metadata.get("ls_model_name"),  # type: ignore
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": "",
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            id = msg.id
                            model_name = metadata.get("ls_model_name")  # type: ignore
                            first_token = False

                        yield dict_to_sse(
                            {
                                "id": msg.id,
                                "created": created,
                                "object": "chat.completion.chunk",
                                "model": metadata.get("ls_model_name"),  # type: ignore
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
            yield dict_to_sse(
                {
                    "id": id,
                    "created": created,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    # if not stream
    else:
        # 非 stream 模式，直接收集所有内容并返回
        response = await app.ainvoke(input=state)
        messages = response.get("messages_with_think", [])
        return {
            "id": response.get("id"),
            "object": "chat.completion",
            "created": created,
            "model": response.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": messages[-1].content,
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": response.get("finish_reason"),
                }
            ],
        }
