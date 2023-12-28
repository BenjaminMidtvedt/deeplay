from typing import (
    List,
    TypeVar,
    overload,
    Literal,
    Optional,
    Any,
)

import torch.nn as nn


from ...module import DeeplayModule
from ...blocks.sequential import SequentialBlock


class TransformAggregateUpdate(SequentialBlock):
    transform: DeeplayModule
    aggregate: DeeplayModule
    update: DeeplayModule
    order: List[str]

    def __init__(
        self,
        transform: DeeplayModule,
        aggregate: DeeplayModule,
        update: DeeplayModule,
        order=["transform", "aggregate", "update"],
        **kwargs: DeeplayModule,
    ):
        super().__init__(
            transform=transform,
            aggregate=aggregate,
            update=update,
            order=order,
            **kwargs,
        )

    @overload
    def configure(
        self,
        order: Optional[List[str]] = None,
        transform: Optional[DeeplayModule] = None,
        aggregate: Optional[DeeplayModule] = None,
        update: Optional[DeeplayModule] = None,
        **kwargs: DeeplayModule,
    ) -> None:
        ...

    @overload
    def configure(self, name: Literal["transform"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["aggregate"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: Literal["update"], *args, **kwargs) -> None:
        ...

    @overload
    def configure(self, name: str, *args, **kwargs: Any) -> None:
        ...

    def configure(self, *args, **kwargs):  # type: ignore
        super().configure(*args, **kwargs)
