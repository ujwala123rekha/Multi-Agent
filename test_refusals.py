
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from utils.prompts import PLANNER_SYSTEM_PROMPT

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["USER_AGENT"] = "TestAgent"
load_dotenv()

async def test_refusal(query):
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    system_prompt = PLANNER_SYSTEM_PROMPT.format(query=query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    print(f"Testing query: {query}")
    try:
        response = await model.ainvoke(messages)
        print(f"Output: {response.content}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    await test_refusal("Tell me about Pond Naravit")
    await test_refusal("Tell me about Aou Thanaboon and Boom Tharatorn")

if __name__ == "__main__":
    asyncio.run(main())
