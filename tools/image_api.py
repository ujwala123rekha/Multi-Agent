import requests
import os
from typing import Optional
import urllib.parse
import base64

# Keywords that trigger the high-quality Flux model.
# Everything else uses the fast default (turbo) model.
COMPLEX_TRIGGERS = [
    # Deities & sacred
    "deity", "god", "goddess", "vishnu", "shiva", "durga", "lakshmi",
    "kali", "ganesh", "ganesha", "rama", "krishna", "hanuman", "saraswati",
    "parvati", "brahma", "indra", "sacred", "divine", "mythological",
    # Multi-person / anatomy
    "multiple people", "crowd", "battle scene", "army", "group of",
    "multi-limb", "many arms", "many hands", "skeleton", "anatomy",
    # Cinematic / detailed
    "cinematic", "8k", "ultra detailed", "hyperrealistic", "unreal engine",
    "epic", "dramatic lighting", "fantasy landscape", "detailed portrait",
]

NEGATIVE_PROMPT = (
    "diagrams, labels, text overlays, arrows, anatomical distortions, "
    "melted faces, extra limbs, fused fingers, blurred features, "
    "low-resolution artifacts, grainy, distorted eyes, text gibberish"
)

class ImageGenerationTool:
    def __init__(self):
        self.art_url = "https://image.pollinations.ai/prompt/"
        self.diagram_url = "https://mermaid.ink/img/"

    def _pick_model(self, prompt: str) -> str:
        """Return 'flux' for complex prompts, 'turbo' for simple ones."""
        p = prompt.lower()
        if any(trigger in p for trigger in COMPLEX_TRIGGERS):
            print(f"[ImageTool] Complex prompt detected → using model=flux")
            return "flux"
        print(f"[ImageTool] Simple prompt detected → using model=turbo")
        return "turbo"

    def generate(self, prompt: str, output_path: str = "generated_image.png") -> Optional[str]:
        """Generate an image or a technical diagram."""
        # Check if the prompt is actually Mermaid code
        if prompt.strip().startswith(("graph", "sequenceDiagram", "flowchart", "classDiagram")):
            return self._generate_diagram(prompt, output_path)
        return self._generate_art(prompt, output_path)

    def _generate_art(self, prompt: str, output_path: str) -> Optional[str]:
        """Generate artistic images. Auto-selects flux for complex prompts."""
        if len(prompt) > 800:
            prompt = prompt[:800]

        model = self._pick_model(prompt)
        encoded_prompt = urllib.parse.quote(prompt)
        encoded_negative = urllib.parse.quote(NEGATIVE_PROMPT)

        url = (
            f"{self.art_url}{encoded_prompt}"
            f"?model={model}&width=1024&height=1024&nologo=true"
            f"&enhance=true&negative_prompt={encoded_negative}"
        )
        return self._download_image(url, output_path)

    def _generate_diagram(self, mermaid_code: str, output_path: str) -> Optional[str]:
        """Generate professional technical diagrams using Mermaid.js."""
        # Clean and encode mermaid code
        # Some renderers prefer base64
        sample_string_bytes = mermaid_code.encode("ascii")
        base64_bytes = base64.b64encode(sample_string_bytes)
        base64_string = base64_bytes.decode("ascii")
        
        url = f"{self.diagram_url}{base64_string}"
        return self._download_image(url, output_path)

    def _download_image(self, url: str, output_path: str) -> Optional[str]:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if "image" not in response.headers.get("Content-Type", ""):
                return None
            with open(output_path, "wb") as f:
                f.write(response.content)
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"Image generation failed: {e}")
            return None
