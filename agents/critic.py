from agents.base import BaseAgent
from config import MODELS


class CriticAgent(BaseAgent):

    def __init__(self, project_name: str):

        super().__init__(
            name="critic",
            model_config=MODELS["critic"],
            project_name=project_name
        )