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
"""
Coupled perturbed Hartree-Fock
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from jax.scipy.sparse.linalg import gmres
from pyscfad.lib import logger

if TYPE_CHECKING:
    from pyscfad.typing import ArrayLike, Array

def solve(
    fvind: Callable[[ArrayLike], ArrayLike],
    mo_energy: ArrayLike,
    mo_occ: ArrayLike,
    h1: ArrayLike,
    s1: ArrayLike | None = None,
    max_cycle: int = 50,
    tol: float = 1e-9,
    hermi: bool = False,
    verbose: Any = logger.WARN,
) -> tuple[Array, None]:
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose)
    else:
        raise NotImplementedError

def solve_nos1(
    fvind: Callable[[ArrayLike], ArrayLike],
    mo_energy: ArrayLike,
    mo_occ: ArrayLike,
    h1: ArrayLike,
    max_cycle: int = 50,
    tol: float = 1e-9,
    hermi: bool = False,
    verbose: Any = logger.WARN,
) -> tuple[Array, None]:
    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = e_a[:,None] - e_i[None,:]

    def vind_vo(mo1):
        v  = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v += e_ai * mo1.reshape(h1.shape)
        return -v.ravel()

    mo1 = gmres(vind_vo, h1.ravel(), tol=tol, maxiter=max_cycle)[0]
    return mo1.reshape(h1.shape), None
