from my_langgraph_agent import env

import json
from my_langgraph_agent.langgraph_agent import app as graph, State
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time


def dict_to_sse(d: dict):
    return "data: " + json.dumps(d) + "\n\n"


app = FastAPI(
    title="API for openwebui-pipelines",
    description="API for openwebui-pipelines",
)


@app.post("/openwebui-pipelines/api")
async def main(inputs: State):
    response = await graph.ainvoke(inputs)
    if env.get("DEBUG"):
        print(response)
    return response


@app.post("/openwebui-pipelines/api/stream")
async def stream(inputs: State):
    async def event_stream():
        start_time = time.time()
        last_time = start_time
        valid_first_token = False
        yield dict_to_sse(
            {
                "event": {
                    "type": "status",
                    "data": {
                        "description": "Warming up...",
                        "done": False,
                    },
                }
            }
        )
        if env.get("DEBUG"):
            print(f"\nReceived inputs: {inputs}\n")
        async for msg, metadata in graph.astream(
            input=inputs, stream_mode="messages"
        ):
            # print(metadata, flush=True)
            if metadata.get("langgraph_node", "") != "agent":
                continue
            if not valid_first_token:
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed > 2:
                    last_time = current_time
                    yield dict_to_sse(
                        {
                            "event": {
                                "type": "status",
                                "data": {
                                    "description": f"Thinking ({(current_time-start_time):.2f}s)",
                                    "done": False,
                                },
                            }
                        }
                    )
            if msg.content:
                if not valid_first_token and (len(msg.content.strip("\n")) == 0):
                    current_time = time.time()
                    elapsed = current_time - last_time
                    if elapsed > 2:
                        last_time = current_time
                        yield dict_to_sse(
                            {
                                "event": {
                                    "type": "status",
                                    "data": {
                                        "description": f"Thinking ({elapsed:.2f}s)",
                                        "done": False,
                                    },
                                }
                            }
                        )
                else:
                    if not valid_first_token:
                        valid_first_token = True
                        current_time = time.time()
                        elapsed = current_time - start_time
                        yield dict_to_sse(
                            {
                                "event": {
                                    "type": "status",
                                    "data": {
                                        "description": f"Done. Total time: {elapsed:.2f} seconds.",
                                        "done": True,
                                    },
                                }
                            }
                        )

                    # 构造 SSE 格式数据
                    yield dict_to_sse(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "role": "assistant",
                                            "content": msg.content,
                                        }
                                    }
                                ]
                            }
                        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def read_root():
    return {"State": "Good"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
