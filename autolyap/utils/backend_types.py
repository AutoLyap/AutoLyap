"""Shared structural typing protocols for optional solver backends."""

from typing import Any, Protocol, Sequence, SupportsFloat, Union

import numpy as np

NumericScalar = Union[int, float, np.floating]


class SupportsStringConversion(Protocol):
    """Protocol for values normalized through ``str(value)``."""

    def __str__(self) -> str:
        ...


class SupportsScalarProduct(Protocol):
    """Protocol for scalar-like symbols participating in backend products."""

    def __mul__(self, other: Any) -> Any:
        ...

    def __rmul__(self, other: Any) -> Any:
        ...


RhoTerm = Union[NumericScalar, SupportsScalarProduct]


class CvxpyStatusModuleProtocol(Protocol):
    """Subset of CVXPY status constants used by AutoLyap."""

    OPTIMAL: str
    OPTIMAL_INACCURATE: str
    INFEASIBLE: str
    INFEASIBLE_INACCURATE: str
    UNBOUNDED: str
    UNBOUNDED_INACCURATE: str


class CvxpyModuleProtocol(CvxpyStatusModuleProtocol, Protocol):
    """Subset of the CVXPY module API used by AutoLyap."""

    def Variable(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def Problem(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def Minimize(self, expr: Any) -> Any:
        ...

    def Parameter(self, *args: Any, **kwargs: Any) -> Any:
        ...


class MosekModelProtocol(Protocol):
    """Protocol for MOSEK Fusion model parameter forwarding."""

    def setSolverParam(self, name: str, value: Any) -> None:
        ...


class MosekLevelHandleProtocol(Protocol):
    """Protocol for solved MOSEK handles exposing ``level()``."""

    def level(self) -> Any:
        ...


class CvxpyValueHandleProtocol(Protocol):
    """Protocol for solved CVXPY handles exposing ``value``."""

    value: Any


ScalarVariableHandle = Union[MosekLevelHandleProtocol, CvxpyValueHandleProtocol]


class MosekExprProtocol(Protocol):
    """Subset of ``mosek.fusion.Expr`` used in helper builders."""

    @staticmethod
    def add(lhs: Any, rhs: Any) -> Any:
        ...

    @staticmethod
    def sub(lhs: Any, rhs: Any) -> Any:
        ...

    @staticmethod
    def hstack(args: Sequence[Any]) -> Any:
        ...

    @staticmethod
    def vstack(args: Sequence[Any]) -> Any:
        ...

    @staticmethod
    def mul(lhs: Any, rhs: Any) -> Any:
        ...


class MosekUpperTriangleVectorProtocol(Protocol):
    """Protocol for MOSEK upper-triangle vector handles."""

    def index(self, idx: int) -> Any:
        ...


class MosekUpperTriangleSolutionHandleProtocol(
    MosekUpperTriangleVectorProtocol,
    MosekLevelHandleProtocol,
    Protocol,
):
    """Protocol for MOSEK upper-triangle handles reused after solve."""


class UpperTriangleValuesProtocol(Protocol):
    """Finite indexable container of numeric upper-triangle entries."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> SupportsFloat:
        ...


class MosekFusionModuleProtocol(Protocol):
    """Subset of MOSEK Fusion module attributes used by AutoLyap."""

    Expr: MosekExprProtocol
    Domain: Any
    ObjectiveSense: Any
    OptimizeError: type[Exception]
    Model: Any
