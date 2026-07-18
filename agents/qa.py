from agents.base import BaseAgent
from config import MODELS


class QAAgent(BaseAgent):

    def __init__(self, project_name: str):

        super().__init__(
            name="qa",
            model_config=MODELS["qa"],
            project_name=project_name,
            allow_search=True
        )