from typing import Literal, Type, Union, Optional, overload
from _typeshed import Incomplete
from deeplay.blocks.conv.conv2d import Conv2dBlock as Conv2dBlock
from deeplay.components.cnn.encdec import ConvolutionalEncoder2d as ConvolutionalEncoder2d
from deeplay.external.layer import Layer as Layer

def resnet(block: Conv2dBlock, stride: int = 1): ...
def resnet18_input(block: Conv2dBlock): ...
def resnet18(encoder: ConvolutionalEncoder2d): ...

class BackboneResnet18(ConvolutionalEncoder2d):
    pool: Layer
    pool_output: Incomplete
    def __init__(self, in_channels: int = 3, pool_output: bool = False) -> None: ...
    @overload
    def style(self, style: Literal["resnet18"], ) -> Self: ...
    @overload
    def style(self, style: Literal["cyclegan_resnet_encoder"], ) -> Self: ...
    @overload
    def style(self, style: Literal["cyclegan_discriminator"], ) -> Self: ...
    @overload
    def style(self, style: Literal["dcgan_discriminator"], ) -> Self: ...
    def style(self, style: str, **kwargs) -> Self: ...
    def forward(self, x): ...
