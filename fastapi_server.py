import json
from langgraph_agent import graph, GraphState, FastAPIState
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import component_helpers

app = FastAPI(
    title="API for openwebui-pipelines",
    description="API for openwebui-pipelines",
    )

@app.post("/openwebui-pipelines/api")
async def main(inputs: FastAPIState):
    response = await graph.ainvoke(inputs)
    print(response)
    return response

@app.post("/openwebui-pipelines/api/stream")
async def stream(inputs: FastAPIState):
    async def event_stream():
        try:
            state: GraphState = {
                "messages": inputs.messages,
                "question": "",  # graph会自动处理 question 和 query，无需手动设置
                "query": "",     # 同上
                "context": [],                        # 初始 context 为空
                "answer": "",                          # 初始 answer 为空
                "chroma_ids": []
            }
            first_token = True
            print(f"\nReceived inputs: {inputs}\n")
            async for msg, metadata in graph.astream(input=state, stream_mode="messages"):
                if msg.content:
                    if metadata.get('langgraph_node', '') != 'generate':
                        continue
                    # 构造 SSE 格式数据
                    data = {
                        "choices": [{
                            "delta": {
                                "role": "assistant",
                                "content": "<think>\n" + msg.content if first_token else msg.content
                            }
                        }]
                    }
                    
                    # 第一行插入 <think>，后续不再插入
                    if first_token:
                        first_token = False
                    
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