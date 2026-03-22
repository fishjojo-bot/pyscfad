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

from functools import partial
from typing import Any, Callable
import numpy
from scipy.sparse.linalg import LinearOperator
from jax.scipy.sparse.linalg import gmres as jax_gmres
from pyscfad.scipy.sparse.linalg import gmres as pyscfad_gmres
from pyscfad.scipy.sparse.linalg import gmres_safe

class GMRESDisp:
    def __init__(self, disp: bool = False) -> None:
        self.disp = disp
        self._iter = 0

    @property
    def iter(self) -> int:
        return self._iter

    def __call__(self, pr_norm: float) -> None:
        self._iter += 1
        if self.disp:
            print(f'gmres cycle {self.iter}: residual norm = {pr_norm}.')

def precond_by_hdiag(h_diag: Any, thresh: float = 1e-12) -> LinearOperator:
    h_diag = numpy.array(h_diag, dtype=float).ravel()
    h_diag[abs(h_diag) < thresh] = 1.
    n = h_diag.size
    M = LinearOperator((n, n), lambda x: x / h_diag)
    return M

def gen_gmres_with_default_kwargs(
    tol: float = 1e-6,
    atol: float = 1e-12,
    maxiter: int = 30,
    x0: Any = None,
    M: Any = None,
    restart: int = 20,
    callback: Any = None,
    callback_type: Any = None,
    safe: bool = False,
    **kwargs: Any,
) -> Callable[..., Any]:
    from pyscfad import config
    if config.moleintor_opt:
        if safe:
            gmres = partial(gmres_safe,
                            tol=tol, atol=atol, maxiter=maxiter,
                            x0=x0, M=M, restart=restart,
                            callback=callback, callback_type=callback_type,
                            **kwargs)
        else:
            gmres = partial(pyscfad_gmres,
                            tol=tol, atol=atol, maxiter=maxiter,
                            x0=x0, M=M, restart=restart,
                            callback=callback, callback_type=callback_type)
    else:
        gmres = partial(jax_gmres,
                        tol=tol, atol=atol, maxiter=maxiter,
                        solve_method='incremental',
                        x0=x0, M=M, restart=restart)
    return gmres

gen_gmres = gen_gmres_with_default_kwargs
