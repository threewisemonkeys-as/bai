from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic

import numpy as np

class ImageState(np.ndarray):
    """Wrapper class for image states"""
    pass

class GridState(list):
    """Wrapper class for grid states"""
    pass

State = ImageState | GridState
StateT = TypeVar("StateT", ImageState, GridState)

Action = int

@dataclass
class Transition(Generic[StateT]):
    state: StateT
    actions: list[Action]
    next_state: StateT


@dataclass
class HypothesisSet:
    hypothesis: list[str] = field(default_factory=list)

    def add(self, h: str):
        self.hypothesis.append(h)

    def append(self, h: str):
        self.hypothesis.append(h)

    def __iter__(self):
        return iter(self.hypothesis)

    def __str__(self):
        return '\n'.join(f"- {h}" for h in self.hypothesis)

class WM:


    def __init__(
        self,
        model: str,
    ):
        self._model = model
        
        
    def _grid_state_check(
        self,
        transition: Transition[GridState],
        hypothesis: HypothesisSet,
    ) -> str | None:
        return None
    
        
    def _image_state_check(
        self,
        transition: Transition[ImageState],
        hypothesis: HypothesisSet,
    ) -> str | None:
        return None

    def _parse_response(
            self,
            response: str
        ) -> bool | None:
        return None

    def __call__(
        self,
        transition: Transition,
        hypothesis: HypothesisSet,
    ) -> bool:

        match transition.state:
            case ImageState():
                response = self._image_state_check(
                    transition,
                    hypothesis,
                )
            case GridState():
                response = self._grid_state_check(
                    transition,
                    hypothesis,
                )
            case _:
                raise TypeError(f"Unknown state type: {type(transition.state)}")

        if response is None:
            raise RuntimeError(f"Error getting response from LLM")

        result = self._parse_response(response)

        if result is None:
            raise RuntimeError(f"Could not parse response -\n{response}")


        return result