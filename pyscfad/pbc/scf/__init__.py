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
Hartree-Fock with PBC
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscfad.pbc.scf import hf
from pyscfad.pbc.scf import khf

if TYPE_CHECKING:
    from pyscfad.pbc.gto import Cell

def RHF(cell: Cell, **kwargs) -> hf.RHF:
    return hf.RHF(cell, **kwargs)

def KRHF(cell: Cell, **kwargs) -> khf.KRHF:
    return khf.KRHF(cell, **kwargs)
