from typing import Annotated, Any, Sequence

from numpydantic import Shape as NPShape
from numpydantic.dtype import Bool, Float64, Int32
from numpydantic.ndarray import NDArray
from pydantic import Field

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(ge=1)]
Shape = Sequence[PositiveInt]
Mask = NDArray[Any, Bool]

RasterWindowShape = NDArray[NPShape['2'], Int32]
RasterFloat64 = NDArray[NPShape['*,*'], Float64]
RasterInt32 = NDArray[NPShape['*,*'], Int32]
RasterBool = NDArray[NPShape['*,*'], Bool]
