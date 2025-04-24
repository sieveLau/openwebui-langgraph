import os
if not os.environ.get("OPENAI_API_KEY"):
    if os.path.exists(".env"):
        env_dict = {}
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    env_dict[key] = value
        api_key = env_dict.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file")
        for key in env_dict:
            os.environ[key] = env_dict[key]
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        raise ValueError("OPENAI_API_KEY is not set and no .env file found")