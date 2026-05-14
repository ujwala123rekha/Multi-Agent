
import asyncio
import os
from dotenv import load_dotenv
from graph import build_graph
from state import AgentState

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["USER_AGENT"] = "TestAgent"
load_dotenv()

async def test_full_pipeline():
    graph = build_graph()
    query = "Tell me about aouboom couple from thailand and show their cute picture"
    
    initial_state = AgentState(
        query=query,
        routing=None,
        research_result=None,
        image_result=None,
        final_output="",
        messages=[]
    )
    
    print(f"Running pipeline for: {query}")
    try:
        result = await graph.ainvoke(initial_state)
        print("\n--- FINAL OUTPUT ---")
        print(result.get("final_output"))
        print("\n--- IMAGE RESULT ---")
        img = result.get("image_result")
        if img:
            print(f"Success: {img.success}")
            print(f"Path: {img.image_path}")
        else:
            print("No image result in state.")
    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
