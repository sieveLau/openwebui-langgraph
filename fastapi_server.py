from init_env import env

import json
from langgraph_agent import app as graph, State
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API for openwebui-pipelines",
    description="API for openwebui-pipelines",
    )

@app.post("/openwebui-pipelines/api")
async def main(inputs: State):
    response = await graph.ainvoke(inputs)
    if env.get('DEBUG'):
        print(response)
    return response

@app.post("/openwebui-pipelines/api/stream")
async def stream(inputs: State):
    async def event_stream():
        try:
            if env.get('DEBUG'): print(f"\nReceived inputs: {inputs}\n")
            async for msg, metadata in graph.astream(input=inputs, stream_mode="messages"):
                if msg.content:
                    if metadata.get('langgraph_node', '') != 'agent':
                        continue
                    # 构造 SSE 格式数据
                    data = {
                        "choices": [{
                            "delta": {
                                "role": "assistant",
                                "content": msg.content
                            }
                        }]
                    }
                    
                    # 编码为 SSE 格式
                    yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            print(f"An error occurred: {e}")

    return StreamingResponse(event_stream(), media_type="application/json")

@app.get("/")
async def read_root():
    return {"State": "Good"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)