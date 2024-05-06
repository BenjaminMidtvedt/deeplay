import deeplay.blocks.base
import nn
import torch.nn.modules.activation
import torch.nn.modules.batchnorm
from _typeshed import Incomplete
from deeplay.blocks.base import BaseBlock as BaseBlock, DeferredConfigurableLayer as DeferredConfigurableLayer
from deeplay.external.layer import Layer as Layer
from deeplay.module import DeeplayModule as DeeplayModule
from deeplay.ops.logs import FromLogs as FromLogs
from deeplay.ops.merge import Add as Add, MergeOp as MergeOp
from deeplay.ops.shape import Permute as Permute
from typing import ClassVar, Literal, Type
from typing_extensions import Self

class Conv2dBlock(deeplay.blocks.base.BaseBlock):
    _style_map: ClassVar[dict] = ...
    def __init__(self, in_channels: int | None, out_channels: int, kernel_size: int = ..., stride: int = ..., padding: int = ..., **kwargs) -> None: ...
    def normalized(self, normalization: Type[nn.Module] | DeeplayModule = ..., mode: str = ..., after: Incomplete | None = ...) -> Self: ...
    def pooled(self, pool: Layer = ..., mode: str = ..., after: Incomplete | None = ...) -> Self: ...
    def upsampled(self, upsample: Layer = ..., mode: str = ..., after: Incomplete | None = ...) -> Self: ...
    def transposed(self, transpose: Layer = ..., mode: str = ..., after: Incomplete | None = ..., remove_upsample: bool = ..., remove_layer: bool = ...) -> Self: ...
    def strided(self, stride: int | tuple[int, ...], remove_pool: bool = ...) -> Self: ...
    def multi(self, n: int = ...) -> Self: ...
    def shortcut(self, merge: MergeOp = ..., shortcut: Literal['auto'] | Type[nn.Module] | DeeplayModule | None = ...) -> Self: ...
def residual(block: Conv2dBlock, order: str = ..., activation: type[torch.nn.modules.activation.ReLU] = ..., normalization: type[torch.nn.modules.batchnorm.BatchNorm2d] = ..., dropout: float = ...): ...
def spatial_self_attention(block: Conv2dBlock, to_channel_last: bool = ..., normalization: Layer | Type[nn.Module] = ...): ...
def spatial_cross_attention(block: Conv2dBlock, to_channel_last: bool = ..., normalization: Layer | Type[nn.Module] = ..., condition_name: str = ...): ...
def spatial_transformer(block: Conv2dBlock, to_channel_last: bool = ..., normalization: Layer | Type[nn.Module] = ..., condition_name: str | None = ...): ...
