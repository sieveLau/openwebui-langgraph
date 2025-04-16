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
    state: GraphState = {
        "messages": inputs.messages,
        "question": "",  # graph会自动处理 question 和 query，无需手动设置
        "query": "",     # 同上
        "context": [],                        # 初始 context 为空
        "answer": "",                          # 初始 answer 为空
        "chroma_ids": []
    }
    async def event_stream():
        first_token = True
        print(f"\nReceived inputs: {inputs}\n")
        async for event in graph.astream(input=state, stream_mode="messages"):
            try:
                if event:
                    if event[1].get('langgraph_node', '') != 'generate':
                        continue
                    if first_token:
                        first_token = False
                        yield "<think>\n"
                    yield event[0].content
            except Exception as e:
                print(f"An error occurred: {e}")

    return StreamingResponse(event_stream(), media_type="application/json")

@app.get("/")
async def read_root():
    return {"State": "Good"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)