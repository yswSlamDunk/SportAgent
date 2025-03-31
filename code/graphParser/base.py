from abc import ABC, abstractmethod
from state import GraphState

class BaseNode(ABC):
    def __init__(self, verbose=True, **kwargs):
        self.name = self.__class__.__name__
        self.verbose = verbose

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        pass

    def log(self, message: str, **kwargs):
        if self.verbose:
            print(f"[{self.name}] {message}")
            for key, value in kwargs.items():
                print(f"{key}: {value}")

    def __call__(self, state: GraphState) -> GraphState:
        return self.execute(state)
        