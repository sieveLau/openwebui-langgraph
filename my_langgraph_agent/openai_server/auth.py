from fastapi import HTTPException, Security
from fastapi.security import api_key
from starlette import status
from my_langgraph_agent import env

api_key_header = api_key.APIKeyHeader(name="Authorization")


async def validate_api_key(key: str = Security(api_key_header)):
    if not key.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )
    token = key[7:]  # 去掉 "Bearer "
    if token != env.get("MY_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return token
