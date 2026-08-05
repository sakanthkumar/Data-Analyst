from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Small interface shared by backend workflow agents."""

    @abstractmethod
    def get_name(self) -> str:
        """Return a stable display/debug name for the agent."""
        raise NotImplementedError
