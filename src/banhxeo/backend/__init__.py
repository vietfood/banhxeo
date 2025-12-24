import importlib
import importlib.util
import math
import os
import tempfile

from banhxeo.backend.torch import TorchInterpreter
from banhxeo.backend.triton import TritonCodegen
from banhxeo.core.buffer import BinaryOp, LazyBuffer, LoadOp, ReduceOp
from banhxeo.utils.helpers import DEBUG


class Backend:
    def schedule(self, buffer: LazyBuffer):
        topo = []
        visited = set()

        def build_topo(v: LazyBuffer):
            if v in visited:
                return

            visited.add(v)

            if v.realized is not None:
                # If a node is realized, it must be part of the topo list so the codegen knows to generate a LoadOp (pointer load) for it.
                topo.append(v)
                return

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

    def is_barrier(self, buf: LazyBuffer):
        # realized buffers are always barriers
        if buf.realized is not None:
            return True

        # if we have a VIEW/CONTIGUOUS op, and its parents is NOT realized (it's a compute op), we realized it
        if isinstance(buf.op, LoadOp):
            if (
                len(buf.src) != 0  # has parents
                and buf.src[0].realized is None
            ):
                return True

        return (
            isinstance(buf.op, ReduceOp)  # specialized kernel
            or buf.op == BinaryOp.MATMUL
        )

    def get_barriers(self, buf: LazyBuffer):
        """
        Traverse up from `buf`.
        If we hit a Barrier, add it to the set and STOP traversing that branch.
        If we hit a non-Barrier (like ADD), keep traversing its parents.
        """
        required = set()
        visited = set()

        def find(node):
            if node in visited:
                return
            visited.add(node)

            # If the node itself is a barrier (and it's NOT the node we are currently trying to exec),
            # then this node must be realized before we can continue.
            if node is not buf and self.is_barrier(node):
                if node.realized is None:
                    required.add(node)
                return

            for parent in node.src:
                find(parent)

        find(buf)
        return required

    def gencode(self, output: LazyBuffer):
        # First we use toposort to linearize the graph
        linear_graph = self.schedule(output)

        if DEBUG >= 2:
            from banhxeo.utils.viz import visualize_schedule_cli

            visualize_schedule_cli(linear_graph)

        # we then generate Triton code
        generator = TritonCodegen(linear_graph)
        src = generator.generate()

        if DEBUG >= 1:
            print(f"--- [DEBUG] GENERATED TRITON KERNEL ({str(output.op).upper()}) ---")
            print(src)
            print("-------------------------------")

        return src, generator

    def exec_elementwise(self, output: LazyBuffer):
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

            for buf, args in generator.input_args.items():
                arg_type = args.type
                metadata = args.metadata

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

    def exec_matmul(self, output: LazyBuffer):
        import triton

        from banhxeo.backend.kernels.matmul import matmul_kernel

        # TODO: Dimension and contiguous check should be in Tensor side instead of Backend side
        assert all([s.realized is not None for s in output.src]), (
            "Source must be realized before Matmul"
        )

        a = output.src[0]
        b = output.src[1]
        c = output

        M, K = a.shape
        K, N = b.shape
        c.allocate()

        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        matmul_kernel[grid](
            a.realized.data,  # type: ignore
            b.realized.data,  # type: ignore
            c.realized.data,  # type: ignore
            M,
            N,
            K,
            a.strides[0],
            a.strides[1],
            b.strides[0],
            b.strides[1],
            c.strides[0],
            c.strides[0],
        )

    def exec_reduce(self, output: LazyBuffer):
        # TODO:
        pass

    def exec(self, output: LazyBuffer):
        if output.realized is not None:
            return output

        # Get all dependencies or barriers should be run first
        barriers = self.get_barriers(output)

        # Then realize all barriers
        for b in barriers:
            if DEBUG >= 2:
                print(f"   [DEBUG] Recursion: Executing dependency {b.op}")
            self.exec(b)

        # HACK: Right now, MatMul and Reduction has specialized kernel
        if output.op == BinaryOp.MATMUL:
            self.exec_matmul(output)
        elif isinstance(output.op, ReduceOp):
            self.exec_reduce(output)
        else:
            self.exec_elementwise(output)
