from rich.console import Console
from rich.logging import RichHandler
import logging

# Configure rich logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("multi_agent")
console = Console()

def log_agent_action(agent_name: str, action: str):
    logger.info(f"[bold blue][{agent_name}][/bold blue] {action}")

def log_error(agent_name: str, error: str):
    logger.error(f"[bold red][{agent_name} Error][/bold red] {error}")
