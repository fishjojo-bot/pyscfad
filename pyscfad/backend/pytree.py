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
from typing import Any

from .config import get_backend

__all__ = [
    'PytreeNode',
    'class_as_pytree_node',
]

def __getattr__(name: str) -> Any:
    return getattr(get_backend(), name)

def class_as_pytree_node(cls: Any, leaf_names: list[str] | tuple[str, ...], num_args: int = 0, exclude_aux_name: tuple[str, ...] = ()) -> Any:
    return get_backend().class_as_pytree_node(cls, leaf_names, num_args=num_args,
                                              exclude_aux_name=exclude_aux_name)
