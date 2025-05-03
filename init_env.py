class _environment:
    def __init__(self):
        self.load()
    def reload(self):
        self.load()
    def load(self):
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
            self.store["TZ"]='Asia/Hong_Kong'
            self.store["DATETIME_FMT"]='%Y-%m-%d %H:%M:%S UTC%z'
            self.store["RAG_DOC_EXPIRE_DAYS"]=7
            self.store["RAG_DOC_CLEAN_DAYS_AGO"]=8

    def get(self, key):
        ret = self.store.get(key)
        if ret is None:
            raise KeyError(f"Key {key} not found in environment")
        return ret
    def get_all(self):
        return self.store.copy()

env = _environment()