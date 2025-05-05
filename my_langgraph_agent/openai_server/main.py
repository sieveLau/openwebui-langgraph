from .authorized_dispatch import router as ad_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(ad_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"message": "OK"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dummy-model-0",
                "object": "model",
                "created": 1686935002,
            },
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8182)
