from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import AgentState, RoutingDecision
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.image_gen import ImageAgent
from agents.base import BaseAgent
from utils.prompts import SYNTHESIZER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Agents
planner = PlannerAgent()
researcher = ResearcherAgent()
image_gen = ImageAgent()
# llama3.2:3b = main model for deep synthesis
synthesizer_llm = BaseAgent(model_name="llama3.2:3b", temperature=0.3)

async def planner_node(state: AgentState) -> Dict[str, Any]:
    decision = await planner.plan(state['query'])
    return {"routing": decision}

async def research_node(state: AgentState) -> Dict[str, Any]:
    if not state['routing'].needs_research:
        return {}
    res = await researcher.research(
        query=state['routing'].optimized_query,
        subtasks=state['routing'].subtasks
    )
    return {"research_result": res}

async def image_node(state: AgentState) -> Dict[str, Any]:
    if not state['routing'].needs_image:
        return {}
    res = await image_gen.generate_image(state['routing'].optimized_query)
    
    if not state['routing'].needs_research:
        if res.success:
            return {"image_result": res, "final_output": "🎨 Here is your generated image!"}
        else:
            return {"image_result": res, "final_output": "❌ Sorry, the image generation API (Pollinations.ai) failed or timed out."}
            
    return {"image_result": res}

async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    query = state['query']
    research = state.get('research_result', "No research conducted.")
    image = state.get('image_result')
    
    # Check if image was generated and successful
    if image and image.success:
        img_info = f"Generated Image available at: {image.image_path}. (Prompt used: {image.prompt})"
    else:
        img_info = "No image was generated for this request."
    
    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT.format(
            query=query, 
            research_result=research, 
            image_result=img_info
        )),
        HumanMessage(content="Synthesize the final response. If an image was generated, mention it and explain what it depicts based on the metadata.")
    ]
    
    response = await synthesizer_llm.ainvoke_llm(messages, tags=["synthesizer"])
    return {"final_output": response.content}

def router_logic(state: AgentState):
    """Conditional edge logic."""
    routing = state['routing']
    destinations = []
    if routing.needs_research:
        destinations.append("research_node")
    if routing.needs_image:
        destinations.append("image_node")
    
    if not destinations:
        return "synthesizer_node"
    
    return destinations

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("image_node", image_node)
    workflow.add_node("synthesizer_node", synthesizer_node)
    
    # Define Edges
    workflow.set_entry_point("planner_node")
    
    # Conditional logic for branching
    workflow.add_conditional_edges(
        "planner_node",
        router_logic,
        {
            "research_node": "research_node",
            "image_node": "image_node",
            "synthesizer_node": "synthesizer_node"
        }
    )
    
    # Fan-in and Conditional logic for nodes
    workflow.add_edge("research_node", "synthesizer_node")
    
    def image_post_logic(state: AgentState):
        # If we also needed research, we must go to synthesizer to combine them
        if state['routing'].needs_research:
            return "synthesizer_node"
        # If this was an IMAGE-ONLY request, we skip synthesizer completely
        return END
        
    workflow.add_conditional_edges(
        "image_node", 
        image_post_logic, 
        {
            "synthesizer_node": "synthesizer_node", 
            END: END
        }
    )
    
    workflow.add_edge("synthesizer_node", END)
    
    return workflow.compile()
