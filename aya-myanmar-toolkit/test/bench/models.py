from dataclasses import dataclass, field
from typing import Any, Optional

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: Optional[float] = None  # top-line metric
    metrics: Optional[dict[str, float]] = None  # other metrics
    htmls: Optional[list[str]] = None  # strings of valid HTML
    convos: Optional[list[MessageList]] = None  # sampled conversations


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: Optional[float] = None
    metrics: dict[str, float] = field(default_factory=dict)
    html: Optional[str] = None
    convo: Optional[MessageList] = None  # sampled conversation


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError