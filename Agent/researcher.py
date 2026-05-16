from agents.base import BaseAgent
from tools.search import SearchTools
from tools.reranker import RerankerTool
from utils.prompts import RESEARCHER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

class ResearcherAgent(BaseAgent):
    def __init__(self):
        # llama3.2:3b = main researcher and retriever model
        super().__init__(model_name="llama3.2:3b", temperature=0)
        self.search_tools = SearchTools()
        self.reranker = RerankerTool()

    async def research(self, query: str, subtasks: list[str]) -> str:
        # 1. Search
        raw_results = await self.search_tools.search_and_fetch(query)
        
        # 2. Rerank
        best_results = self.reranker.rerank(query, raw_results)
        
        # 3. Format for LLM
        context = "\n\n".join([
            f"SOURCE: {r['url']}\nCONTENT: {r['content']}" 
            for r in best_results
        ])
        
        # 4. Synthesize report
        formatted_subtasks = "\n".join([f"- {task}" for task in subtasks])
        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT.format(results=context, query=query, subtasks=formatted_subtasks)),
            HumanMessage(content="Please synthesize the research findings into a structured report.")
        ]
        
        response = await self.ainvoke_llm(messages, tags=["researcher", "llama3.2:3b"])
        return response.content
