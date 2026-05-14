import gradio as gr
import asyncio
import os
from PIL import Image
from graph import build_graph
from state import AgentState

# Load the graph
graph = build_graph()

async def run_pipeline_ui(query: str):
    """Bridge between Gradio and the LangGraph pipeline."""
    initial_state = AgentState(
        query=query,
        routing=None,
        research_result=None,
        image_result=None,
        final_output="",
        messages=[]
    )
    
    # Use ainvoke to get the final result directly
    result = await graph.ainvoke(initial_state)
    
    report = result.get("final_output", "No output generated.")
    image_data = result.get("image_result")
    
    # Load the image as a PIL object so Gradio renders it on screen reliably
    pil_image = None
    if image_data and image_data.success and image_data.image_path:
        try:
            pil_image = Image.open(image_data.image_path)
        except Exception as e:
            print(f"Warning: Could not load image from {image_data.image_path}: {e}")
    
    return report, pil_image

def launch_ui():
    with gr.Blocks(title="Multi-Agent AI System") as demo:
        gr.Markdown("# 🚀 Graph-Based Multi-Agent AI System")
        gr.Markdown("Built with **LangGraph**, **LangChain**, and **Ollama**. (Online for Research & Images)")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="User Query", 
                    placeholder="Search for anything or images...",
                    lines=3
                )
                submit_btn = gr.Button("Execute Pipeline", variant="primary")
            
        with gr.Row():
            with gr.Column(scale=3):
                output_report = gr.Markdown(label="Synthesized Report")
            with gr.Column(scale=2):
                output_image = gr.Image(label="Generated Visual")
        
        submit_btn.click(
            fn=run_pipeline_ui,
            inputs=input_text,
            outputs=[output_report, output_image]
        )

    demo.launch(theme=gr.themes.Soft(), share=False)

if __name__ == "__main__":
    launch_ui()
