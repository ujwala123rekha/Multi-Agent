import asyncio
import sys
from graph import build_graph
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
from utils.logging import logger, console, log_agent_action

import os

# SILENCE LangSmith if tracing is disabled
if os.getenv("LANGCHAIN_TRACING_V2") == "false":
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ.pop("LANGCHAIN_API_KEY", None)

# Set a default User-Agent for web requests
os.environ["USER_AGENT"] = "MultiAgentApp/1.0"

load_dotenv()

console = Console()

async def run_pipeline(query: str):
    graph = build_graph()
    
    initial_state = {
        "query": query,
        "messages": []
    }
    
    console.print(f"\n[bold blue]>>> Processing Query:[/bold blue] {query}")
    
    async for event in graph.astream(initial_state):
        for node_name, output in event.items():
            if node_name == "planner_node":
                routing = output.get("routing")
                log_agent_action("Planner", "Decision formulated.")
                reasoning_str = "\n".join([f"- {r}" for r in routing.reasoning]) if isinstance(routing.reasoning, list) else routing.reasoning
                console.print(Panel(
                    f"Research: [bold]{routing.needs_research}[/bold]\n"
                    f"Image: [bold]{routing.needs_image}[/bold]\n"
                    f"Reasoning:\n{reasoning_str}",
                    title="[bold green]Planner Decision[/bold green]",
                    expand=False
                ))
            elif node_name == "research_node":
                log_agent_action("Researcher", "Deep research completed.")
            elif node_name == "image_node":
                log_agent_action("ImageGen", "Visual asset generated.")
            elif node_name == "synthesizer_node":
                final_output = output.get("final_output")
                console.print("\n[bold cyan]─── FINAL RESPONSE ───[/bold cyan]")
                console.print(Markdown(final_output))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is LangGraph and how does it compare to LangChain agents?"
        
    asyncio.run(run_pipeline(query))
