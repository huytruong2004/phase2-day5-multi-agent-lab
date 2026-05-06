"""Domain-specific errors for the lab."""


class LabError(Exception):
    """Base error for the lab package."""


class StudentTodoError(LabError):
    """Raised where learners are expected to implement core logic."""


class AgentExecutionError(LabError):
    """Raised when an agent fails after retries/fallbacks."""


class AgentInputError(LabError):
    """Raised when an agent's input preconditions are not met."""


class ValidationError(LabError):
    """Raised when state or output validation fails."""


class WorkflowTimeoutError(LabError):
    """Raised when the workflow exceeds the configured wall-clock budget."""
