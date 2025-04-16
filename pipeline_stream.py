import requests
from typing import List, Union, Generator, Iterator
try:
    from pydantic.v1 import BaseModel
except Exception:
    from pydantic import BaseModel

class Pipeline:

    # TODO: Add google PSE id and key, tavily key
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.id = "LangGraph Agent (Stream)"
        self.name = "LangGraph Agent (Stream)"

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
            ) -> Union[str, Generator, Iterator]:

        url = 'http://127.0.0.1:8082/openwebui-pipelines/api/stream'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {
            "messages": [[msg['role'], msg['content']] for msg in messages]
        }
        print("data",data)
        
        try:
            response = requests.post(url, json=data, headers=headers, stream=True)
            response.raise_for_status()
        except requests.RequestException as err:
            # 捕获所有网络相关错误，包括连接失败、超时、HTTP错误等
            print(f"Request error occurred: {err}")
            print(f"Response body: {getattr(err.response, 'text', 'No response')}")
            return "Failed"
        return response.iter_lines()