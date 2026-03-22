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

import jax
from jax import numpy as jnp

def is_array(x: Any) -> bool:
    return isinstance(x, jax.Array)

def to_numpy(x: Any) -> Any:
    if is_array(x):
        x = jax.lax.stop_gradient(x)
        if is_array(x):
            x = x.__array__()
    return x

def vmap(fun: Callable[..., Any], in_axes: Any = 0, out_axes: Any = 0, chunk_size: int | None = None, signature: str | None = None) -> Any:
    return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

# TODO deprecate these
def index_update(x: Any, idx: Any, y: Any) -> Any:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].set(y)

def index_add(x: Any, idx: Any, y: Any) -> Any:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].add(y)

def index_mul(x: Any, idx: Any, y: Any) -> Any:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].multiply(y)


