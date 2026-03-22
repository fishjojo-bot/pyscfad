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
Hartree-Fock theory
"""
from __future__ import annotations
from typing import Any

from pyscfad.scf import hf
from pyscfad.scf import uhf
from pyscfad.scf import rohf

def RHF(mol: Any, **kwargs: Any) -> hf.RHF:
    return hf.RHF(mol, **kwargs)

def UHF(mol: Any, **kwargs: Any) -> uhf.UHF:
    return uhf.UHF(mol, **kwargs)

def ROHF(mol: Any, **kwargs: Any) -> rohf.ROHF:
    return rohf.ROHF(mol, **kwargs)
