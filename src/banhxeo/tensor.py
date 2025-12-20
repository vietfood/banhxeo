import importlib
import importlib.util
import math
import os
import tempfile
from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from src.banhxeo.buffer import BinaryOps, LazyBuffer, LoadOps, UnaryOps
from src.banhxeo.codegen import TritonCodegen
from src.banhxeo.helpers import DEBUG


class Tensor:
    __slots__ = "lazydata", "requires_grad"
    __deletable__ = ("_ctx",)
    training: ClassVar[bool] = False

    def __init__(
        self,
        data: Optional[
            Union[LazyBuffer, List, Tuple, np.ndarray, torch.Tensor, int, float]
        ] = None,
    ):
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        elif isinstance(data, (int, float)):
            self.lazydata = LazyBuffer(LoadOps.CONST, args=[data])
        elif isinstance(data, (List, Tuple, np.ndarray)):
            self.lazydata = LazyBuffer(LoadOps.FROM_CPU, args=[data])

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_lazy = self.lazydata.build(BinaryOps.ADD, other.lazydata)
        return Tensor(out_lazy)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.lazydata.build(BinaryOps.MUL, other.lazydata))

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_lazy = self.lazydata.build(BinaryOps.SUB, other.lazydata)
        return Tensor(out_lazy)

    def log(self):
        return Tensor(self.lazydata.build(UnaryOps.LOG2))

    def exp(self):
        return Tensor(self.lazydata.build(UnaryOps.EXP2))

    def sin(self):
        return Tensor(self.lazydata.build(UnaryOps.SIN))

    def sqrt(self):
        return Tensor(self.lazydata.build(UnaryOps.SQRT))

    def schedule(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.src:
                    build_topo(child)
                topo.append(v)

        build_topo(self.lazydata)
        return topo

    # This is where we actually run the compiler!
    def realize(self):
        # First we use toposort to linearize the graph
        linear_graph = self.schedule()

        # we then generate Triton code
        codegen = TritonCodegen(linear_graph)
        src = codegen.generate()

        if DEBUG >= 1:
            print("--- GENERATED TRITON KERNEL ---")
            print(src)
            print("-------------------------------")

        # Write to a temporary file so Triton can inspect the source
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import torch\n")
            f.write("import triton\n")
            f.write("import triton.language as tl\n\n")
            f.write(src)
            temp_path = f.name

        try:
            # Dynamically import the generated module
            spec = importlib.util.spec_from_file_location("generated_kernel", temp_path)
            if spec is None:
                raise ValueError("Generated kernel is None")

            mod = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ValueError("Loader is None")

            spec.loader.exec_module(mod)
            triton_kernel = mod.generated_kernel
        finally:
            # Clean up the temp file
            os.remove(temp_path)

        # prepare input tensors
        input_tensors = []
        N = 0
        for buf in codegen.schedule:
            if buf.op == LoadOps.FROM_CPU:
                t = torch.tensor(buf.args[0], dtype=torch.float32, device="cuda")
                input_tensors.append(t)
                N = len(t)
            elif buf.op == LoadOps.CONST:
                t = torch.full(
                    buf.args[1], buf.args[0], dtype=torch.float32, device="cuda"
                )
                input_tensors.append(t)
                N = len(t)

        self.lazydata.realized = torch.empty(N, device="cuda", dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (math.ceil(N / BLOCK_SIZE),)
        triton_kernel[grid](
            *input_tensors, self.lazydata.realized, N, BLOCK_SIZE=BLOCK_SIZE
        )
        return self.lazydata.realized
