
import asyncio
import os
from dotenv import load_dotenv

# Set these before any imports to avoid LangSmith errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["USER_AGENT"] = "TestAgent"

load_dotenv()

async def test_planner():
    from agents.planner import PlannerAgent
    print("Initializing Planner...")
    planner = PlannerAgent()
    query = "Tell me about aouboom couple from thailand and show their cute picture"
    
    print(f"Testing Planner with query: {query}")
    try:
        decision = await planner.plan(query)
        print("Planner Decision Success!")
        print(f"Needs Research: {decision.needs_research}")
        print(f"Needs Image: {decision.needs_image}")
        print(f"Optimized Query: {decision.optimized_query}")
        return decision
    except Exception as e:
        print(f"Planner Error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_researcher(query):
    from agents.researcher import ResearcherAgent
    print("\nInitializing Researcher...")
    researcher = ResearcherAgent()
    print(f"Testing Researcher with query: {query}")
    try:
        res = await researcher.research(query)
        print("Researcher Success!")
        print(f"Result (first 200 chars): {res[:200]}...")
    except Exception as e:
        print(f"Researcher Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    decision = await test_planner()
    if decision and decision.needs_research:
        await test_researcher(decision.optimized_query)

if __name__ == "__main__":
    asyncio.run(main())
