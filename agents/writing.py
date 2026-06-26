from agents.base import BaseAgent
from config import MODELS


class WritingAgent(BaseAgent):

    def __init__(self, project_name: str):

        super().__init__(
            name="writing",
            model_config=MODELS["writing"],
            project_name=project_name
        )