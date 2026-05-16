import json
from agents.base import BaseAgent
from state import RoutingDecision
from utils.prompts import PLANNER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class PlannerAgent(BaseAgent):
    def __init__(self):
        # Using a small, fast model for routing and planning
        super().__init__(model_name="llama3.2:1b", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=RoutingDecision)

    async def plan(self, query: str) -> RoutingDecision:
        system_prompt = PLANNER_SYSTEM_PROMPT.format(query=query)
        format_instructions = self.parser.get_format_instructions()
        
        messages = [
            SystemMessage(content=f"{system_prompt}\n{format_instructions}"),
            HumanMessage(content=query)
        ]
        
        response = await self.ainvoke_llm(messages, tags=["planner", "llama3.2:3b"])
        # Some models might return the JSON wrapped in markdown or just the string
        content = response.content
        
        # Robust JSON extraction
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Find the first { and last }
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            return self.parser.parse(content)
        except Exception as e:
            # If parsing fails, create a default decision to avoid crashing the whole pipeline
            return RoutingDecision(
                needs_research=True,
                needs_image="picture" in query.lower() or "image" in query.lower(),
                optimized_query=query,
                subtasks=["Process query as general research"],
                reasoning=["Fallback decision due to planner output format issues."]
            )
