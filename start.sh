python ./langgraph_agent.py
uvicorn fastapi_server:app --reload --port 8082 --log-level debug
