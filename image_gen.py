from agents.base import BaseAgent
from tools.image_api import ImageGenerationTool
from state import ImageResult
from utils.prompts import IMAGE_PROMPT_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
import uuid
import os

class ImageAgent(BaseAgent):
    def __init__(self):
        # Using a fast, small model just to expand the text prompt
        super().__init__(model_name="llama3.2:1b", temperature=0.7) 
        self.tool = ImageGenerationTool()

    async def generate_image(self, user_prompt: str) -> ImageResult:
        # 1. Optimize prompt
        messages = [
            SystemMessage(content=IMAGE_PROMPT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        response = await self.ainvoke_llm(messages, tags=["image_gen", "prompt_optimizer"])
        optimized_prompt = response.content.strip()
        
        # Guard against LLM safety refusals being used as prompts
        refusal_keywords = ["sorry", "i can't", "i cannot", "my purpose", "hate speech", "policy"]
        if any(keyword in optimized_prompt.lower() for keyword in refusal_keywords):
            # Fallback to the original prompt if the LLM refuses
            optimized_prompt = user_prompt
        
        # 2. Generate
        img_id = str(uuid.uuid4())[:8]
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"img_{img_id}.png")
        
        image_path = self.tool.generate(optimized_prompt, output_path=path)
        
        if image_path:
            return ImageResult(image_path=image_path, prompt=optimized_prompt, success=True)
        else:
            return ImageResult(image_path="", prompt=optimized_prompt, success=False)

