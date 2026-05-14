from typing import List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class RoutingDecision(BaseModel):
    """Schema for the initial planning and routing decision."""
    needs_research: bool = Field(description="Whether the query requires external web research.")
    needs_image: bool = Field(description="Whether the query requires image generation.")
    optimized_query: str = Field(description="The user query rewritten for better downstream performance.")
    subtasks: List[str] = Field(description="A list of sub-steps to achieve the goal.")
    reasoning: List[str] = Field(description="Step-by-step reasoning for this routing decision.")

class ImageResult(BaseModel):
    """Schema for the result of an image generation task."""
    image_path: str = Field(description="Local path to the generated image file.")
    prompt: str = Field(description="The actual prompt used to generate the image.")
    success: bool = Field(description="Whether the generation was successful.")

class AgentState(TypedDict):
    """The overall state of the multi-agent system."""
    query: str
    routing: Optional[RoutingDecision]
    research_result: Optional[str]
    image_result: Optional[ImageResult]
    final_output: Optional[str]
    # We use Annotated with operator.add to append messages to the list
    messages: Annotated[List[BaseMessage], operator.add]
