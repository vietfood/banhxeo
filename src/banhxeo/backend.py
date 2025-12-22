import importlib
import importlib.util
import math
import os
import tempfile

from banhxeo.buffer import LazyBuffer
from banhxeo.codegen import TorchInterpreter, TritonCodegen
from banhxeo.helpers import DEBUG


class Backend:
    def schedule(self, buffer: LazyBuffer):
        topo = []
        visited = set()

        def build_topo(v: LazyBuffer):
            if v in visited:
                return

            visited.add(v)

            if v.realized is not None: 
                return # stop when see realized buffer

            for child in v.src:
                build_topo(child)
            topo.append(v)

        build_topo(buffer)
        return topo

    def exec(self, output: LazyBuffer):
        """
        Exec should modify output LazyBuffer data directly
        """
        raise NotImplementedError


class CPUBackend(Backend):
    def exec(self, output: LazyBuffer):
        linear_graph = self.schedule(output)
        # we then interpret to Pytorch directly (instead of compiling)
        interpreter = TorchInterpreter(linear_graph)
        interpreter.run()


class CUDABackend(Backend):
    """
    Use TritonCodegen
    """
    def is_barrier(buf: LazyBuffer):
        # realized buffers are always barriers
        if buf.realized is not None:
            return True

        return (
            isinstance(buf.op, ReduceOp)   # Sum/Max always start a new reduction kernel
            or buf.op == BinaryOp.MATMUL   # Matmul is a specialized kernel
            or buf.op == LoadOp.CONTIGUOUS # Contiguous is a memory copy kernel
        )

    def gencode(self, output: LazyBuffer):
        # First we use toposort to linearize the graph
        linear_graph = self.schedule(output)

        # we then generate Triton code
        generator = TritonCodegen(linear_graph)
        src = generator.generate()

        if DEBUG >= 1:
            print("--- GENERATED TRITON KERNEL ---")
            print(src)
            print("-------------------------------")

        return src, generator

    def exec(self, output: LazyBuffer):
        src, generator = self.gencode(output)

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

            # prepare input tensors
            input_tensors = []
            N = 0
            for arg in generator.input_args:
                buf = arg.buf
                arg_type = arg.type
                metadata = arg.metadata

                if arg_type == "ptr":
                    buf.allocate()
                    input_tensors.append(buf.realized.data)
                elif arg_type == "const":
                    # append const directly
                    input_tensors.append(buf.args[0])

                if metadata is not None:
                    for m in metadata:
                        if m == "shape":
                            input_tensors.extend(buf.view.shape)
                        elif m == "stride":
                            input_tensors.extend(buf.view.strides)

            output.allocate()
            assert output.realized is not None  # should be None after allocated

            N = math.prod(output.view.shape)  # total size
            BLOCK_SIZE = 1024  # hardcode block size
            grid = (math.ceil(N / BLOCK_SIZE),)
            triton_kernel[grid](
                *input_tensors, output.realized.data, N, BLOCK_SIZE=BLOCK_SIZE
            )
        finally:
            # Clean up the temp file
            os.remove(temp_path)
