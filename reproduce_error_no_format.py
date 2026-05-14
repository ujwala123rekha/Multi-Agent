
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from utils.prompts import PLANNER_SYSTEM_PROMPT
from langchain_core.output_parsers import PydanticOutputParser
from state import RoutingDecision

# Set these before any imports to avoid LangSmith errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["USER_AGENT"] = "TestAgent"

load_dotenv()

async def test_planner_raw_no_format():
    # Remove format="json" to see raw text
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    query = "Tell me about aouboom couple from thailand and show their cute picture"
    
    parser = PydanticOutputParser(pydantic_object=RoutingDecision)
    format_instructions = parser.get_format_instructions()
    system_prompt = PLANNER_SYSTEM_PROMPT.format(query=query)
    
    messages = [
        SystemMessage(content=f"{system_prompt}\n{format_instructions}"),
        HumanMessage(content=query)
    ]
    
    print(f"Calling Ollama (NO FORMAT) with query: {query}")
    try:
        response = await model.ainvoke(messages)
        print("--- RAW LLM OUTPUT ---")
        print(response.content)
        print("-----------------------")
    except Exception as e:
        print(f"Ollama Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_planner_raw_no_format())
