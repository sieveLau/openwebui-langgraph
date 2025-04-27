class _environment:
    def __init__(self):
        self.store = {}
        import os
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"')
                        self.store[key] = value
            env_api_key = os.environ.get("OPENAI_API_KEY")
            if env_api_key:
                self.store["OPENAI_API_KEY"] = env_api_key
            api_key = self.store.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is missing")
            if 'DEBUG' not in self.store:
                self.store['DEBUG'] = False

    def get(self, key):
        ret = self.store.get(key)
        if ret is None:
            raise KeyError(f"Key {key} not found in environment")
        return ret

env = _environment()