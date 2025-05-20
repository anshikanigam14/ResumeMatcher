import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_llm():
    return ChatOpenAI( model="gpt-4o", temperature=0)