import hashlib
import linecache
import math
import time

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

    kernel_cache = dict()
    kernel_file_name = dict()

    def compile_triton_src(self, src_code: str):
        """
        Compiles Triton source code from string without touching disk.
        How we do this (Thanks Gemini for the solution):
        * Create a virtual file (but inject directly in to linecache) so we can trick Python to think we have "this file"
        * Then we can use the same method exec (faster than writing to disk)
        * We need this because we can't run JIT compiled Triton kernel directly with exec (because @triton.jit() use inspect.getsource() first)
        """

        # clear cache
        if len(linecache.cache) > 10000:
            for name in self.kernel_file_name.values():
                linecache.checkcache(name)

        import triton

        # This acts as the cache key and the virtual filename
        src_hash = hashlib.sha1(src_code.encode("utf-8")).hexdigest()
        virtual_filename = f"<triton_kernel_{src_hash}.py>"

        # Inject source into linecache
        # inspect.getsource() looks here before checking disk
        # Format: (size, mtime, lines, fullname)
        linecache.cache[virtual_filename] = (
            len(src_code),
            None,
            src_code.splitlines(keepends=True),
            virtual_filename,
        )

        # Create the execution context
        context = {
            "triton": triton,
            "tl": triton.language,
        }

        # Compile the string into a code object
        bytecode = compile(src_code, virtual_filename, "exec")

        # Execute bytecode to define the function in 'context'
        exec(bytecode, context)

        # Retrieve the JIT-ed function
        kernel = context["generated_kernel"]

        return kernel, virtual_filename

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
        src_hash = hashlib.sha1(src.encode("utf-8")).hexdigest()

        if src not in self.kernel_cache.keys():
            try:
                kernel, kernel_name = self.compile_triton_src(src)
                self.kernel_cache[src_hash] = kernel
                self.kernel_file_name[id(kernel)] = kernel_name
            except Exception as e:
                print(f"FAILED SOURCE:\n{src}")
                raise e

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
        grid = lambda META: (math.ceil(N / META["BLOCK_SIZE"]),)  # noqa: E731

        st = 0
        if DEBUG >= 3:
            import torch

            torch.cuda.synchronize()
            st = time.perf_counter()

        self.kernel_cache[src_hash][grid](*input_tensors, output.realized.data, N)

        if DEBUG >= 3:
            import torch

            torch.cuda.synchronize()
            et = time.perf_counter()
            print(f"[DEBUG] KERNEL {output.op} took {(et - st) * 1000:.2f} ms")

    def exec_matmul(self, output: LazyBuffer):
        import triton

        from banhxeo.backend.kernels.matmul import matmul_kernel

        for s in output.src:
            # We need to realize source before matmul kernel
            # Currently matmul is a different kernel
            self.exec(s)

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
            c.strides[1],
        )

    def exec_reduce(self, output: LazyBuffer):
        from banhxeo.backend.kernels.reduce import reduce_max_kernel, reduce_sum_kernel
        from banhxeo.core.buffer import ReduceOp

        kernel_map = {ReduceOp.SUM: reduce_sum_kernel, ReduceOp.MAX: reduce_max_kernel}

        src = output.src[0]

        # If we permuted dimensions to get here, the strides are messy.
        # Making it contiguous ensures stride_row = N and stride_col = 1
        if not src.view.is_contiguous():
            src = src.contiguous()
            self.exec(src)
        else:
            self.exec(src)

        # Input is logically (Batch..., N) -> (M, N)
        *batch_dims, N = src.shape
        M = math.prod(batch_dims)

        output.allocate()

        grid = (M,)

        # Strides: Since we forced contiguous, we know the layout
        # (M, N) contiguous layout:
        # stride_row = N (jump N elements to get to next row)
        # stride_col = 1 (elements in row are sequential)
        kernel_map[output.op][grid](  # type: ignore
            src.realized.data,  # type: ignore
            output.realized.data,  # type: ignore
            stride_x_row=N,
            stride_x_col=1,
            N=N,
            BLOCK_SIZE=1024,
        )

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

        if DEBUG >= 2:
            print(output.realized.data)  # type: ignore
