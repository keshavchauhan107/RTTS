import os
from dotenv import load_dotenv

# Load all .env variables once
load_dotenv()

def get_env_value(key):
    return os.getenv(key)
