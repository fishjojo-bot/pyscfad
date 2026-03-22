# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Any, Callable

import math

def stop_gradient(x: Any) -> Any:
    return x

class custom_jvp:
    """Fake ``custom_jvp`` that does nothing.
    """
    def __init__(self, fun: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.fun = fun
        self.jvp = None

    def defjvp(self, jvp: Callable[..., Any]) -> Callable[..., Any]:
        self.jvp = jvp
        return jvp

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fun(*args, **kwargs)

def jit(fun: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    return fun

def pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: Any = None,
    vmap_method: Any = None,
    **kwargs: Any,
) -> Any:
    return callback(*args, **kwargs)

def while_loop(
    cond_fun: Callable[[Any], Any],
    body_fun: Callable[[Any], Any],
    init_val: Any,
) -> Any:
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

# TODO deprecate these
class _Indexable(object):
    # pylint: disable=line-too-long
    """
    see https://github.com/google/jax/blob/97d00584f8b87dfe5c95e67892b54db993f34486/jax/_src/ops/scatter.py#L87
    """
    __slots__ = ()

    def __getitem__(self, idx: Any) -> Any:
        return idx

index = _Indexable()

def index_update(x: Any, idx: Any, y: Any) -> Any:
    x[idx] = y
    return x

def index_add(x: Any, idx: Any, y: Any) -> Any:
    x[idx] += y
    return x

def index_mul(x: Any, idx: Any, y: Any) -> Any:
    x[idx] *= y
    return x

class ShapeDtypeStruct:
    def __init__(self, shape: tuple[int, ...], dtype: Any, **kwargs: Any) -> None:
        self.shape = shape
        self.dtype = dtype

    size = property(lambda self: math.prod(self.shape))
    ndim = property(lambda self: len(self.shape))
