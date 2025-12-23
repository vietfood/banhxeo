from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

from banhxeo.buffer import BinaryOp, LazyBuffer, LoadOp, UnaryOp


@dataclass
class InputArgument:
    buf: LazyBuffer
    type: Optional[Literal["ptr", "const"]] = None
    # shape, stride
    metadata: Optional[List[str]] = None


class TorchInterpreter:
    def __init__(self, schedule: List[LazyBuffer]):
        self.schedule = schedule

    def run(self):
        for buf in self.schedule:
            # allodate all buffers (because this is interpreter)
            buf.allocate()
            assert buf.realized is not None, "Allocation failed"
            if isinstance(buf.op, LoadOp):
                # CONST, DEFAULT, FROM_CPU are already handled by allocate() implicitly.
                if buf.op == LoadOp.VIEW:
                    assert buf.src[0].realized is not None
                    # we only change shape and stride of current data
                    buf.realized.data = buf.src[0].realized.data.as_strided(
                        size=buf.shape,
                        stride=buf.strides,
                        storage_offset=buf.offset,
                    )
                elif buf.op == LoadOp.CONTIGUOUS:
                    assert buf.src[0].realized is not None
                    buf.realized.data = buf.src[0].realized.data.contiguous()
            elif isinstance(buf.op, BinaryOp):
                # as it should be
                assert buf.src[0].realized is not None
                assert buf.src[1].realized is not None
                op_map = {
                    BinaryOp.ADD: torch.add,
                    BinaryOp.SUB: torch.sub,
                    BinaryOp.MUL: torch.mul,
                    BinaryOp.MATMUL: torch.matmul,
                }
                buf.realized.data = op_map[buf.op](
                    buf.src[0].realized.data, buf.src[1].realized.data
                )
            elif isinstance(buf.op, UnaryOp):
                assert buf.src[0].realized is not None
                op_map = {
                    UnaryOp.LOG2: torch.log2,
                    UnaryOp.EXP2: torch.exp2,
                    UnaryOp.SIN: torch.sin,
                    UnaryOp.SQRT: torch.sqrt,
                }
                buf.realized.data = op_map[buf.op](buf.src[0].realized.data)


class TritonCodegen:
    def __init__(self, schedule: List[LazyBuffer]):
        # linearize LazyBuffer graph
        self.schedule = schedule
        # Map LazyBuffer to Variable name
        self.var_names = dict()
        # Keep track of which input pointers we need
        self.input_args = []
        # the gen code
        self.code = []

    def get_var_name(self, buf: LazyBuffer) -> str:
        if buf not in self.var_names:
            # If it's a LoadOp, it needs a variable name that corresponds to a kernel argument
            if isinstance(buf.op, LoadOp):
                name = f"in_{len(self.input_args)}"
                if buf.op == LoadOp.FROM_CPU or buf.op == LoadOp.DEFAULT:
                    # assume LoadOp.FROM_CPU is linearly
                    # note that DEFAULT create an empty Tensor so we assume it is a pointer too
                    self.input_args.append(InputArgument(buf, "ptr"))
                elif buf.op == LoadOp.CONST:
                    self.input_args.append(InputArgument(buf, "const"))
                elif buf.op == LoadOp.VIEW:
                    self.input_args.append(
                        InputArgument(buf, None, ["shape", "stride"])
                    )
                elif buf.op == LoadOp.FROM_NUMPY:
                    self.input_args.append(
                        InputArgument(buf, "ptr", ["shape", "stride"])
                    )
                self.var_names[buf] = name
            else:
                self.var_names[buf] = f"temp_{len(self.var_names)}"
        return self.var_names[buf]

    def render_indexing(self, buf):
        name = self.get_var_name(buf)
        src_name = self.get_var_name(buf.src[0])
        code = [f"    {name}_offset = {buf.view.offset}", f"    {name}_idx = temp_idx"]
        for i in reversed(range(len(buf.view.shape))):
            # idx_i = temp_idx % shape_i
            # temp_idx = temp_idx // shape_i
            # offset += idx_i * stride_i
            code.extend(
                [
                    f"    {name}_idx_{i} = {name}_idx % {name}_shape_{i}",
                    f"    {name}_idx = {name}_idx // {name}_shape_{i}",
                    f"    {name}_offset += {name}_idx_{i} * {name}_stride_{i}",
                ]
            )
        return chr(10).join(
            [
                *code,
                f"    {name} = tl.load({src_name}_ptr + {name}_offset)",
            ]
        )

    def should_materialize(self, buf):
        # Count children (buffers that use this as src)
        children = [b for b in self.schedule if buf in b.src]

        # Only materialize if it has multiple children or non-VIEW children
        if len(children) == 1 and children[0].op == LoadOp.VIEW:
            return False

        return True

    def generate(self):
        # define the body and identify inputs
        body_code = []
        for buf in self.schedule:
            name = self.get_var_name(buf)

            if isinstance(buf.op, LoadOp):
                if self.should_materialize(buf):
                    if buf.op == LoadOp.CONST:
                        # hardcode value
                        body_code.append(f"    {name} = {name}_const")
                    elif buf.op == LoadOp.FROM_CPU:
                        body_code.append(
                            f"    {name} = tl.load({name}_ptr + linear_offsets, mask=linear_mask)"
                        )
                    elif buf.op == LoadOp.VIEW:
                        body_code.append(self.render_indexing(buf))
            elif isinstance(buf.op, BinaryOp):
                src0 = self.get_var_name(buf.src[0])
                src1 = self.get_var_name(buf.src[1])
                op_map = {
                    BinaryOp.ADD: "+",
                    BinaryOp.SUB: "-",
                    BinaryOp.MUL: "*",
                }
                body_code.append(f"    {name} = {src0} {op_map[buf.op]} {src1}")
            elif isinstance(buf.op, UnaryOp):
                src0 = self.get_var_name(buf.src[0])
                op_map = {
                    UnaryOp.LOG2: "tl.log2",
                    UnaryOp.EXP2: "tl.exp2",
                    UnaryOp.SIN: "tl.sin",
                    UnaryOp.SQRT: "tl.sqrt",
                }
                body_code.append(f"    {name} = {op_map[buf.op]}({src0})")

        args_sig = []
        for args in self.input_args:
            buf = args.buf
            sig = args.type
            metadata = args.metadata

            if sig is not None:
                args_sig.append(f"{self.get_var_name(buf)}_{sig}")

            if metadata is not None:
                for m in metadata:
                    args_sig.extend(
                        [
                            f"{self.get_var_name(buf)}_{m}_{i}"
                            for i in range(len(buf.view.shape))
                        ]
                    )  # note that we assume shape len always equals to strides len

        kernel_def = [
            "@triton.jit",
            f"def generated_kernel({', '.join(args_sig)}, out_ptr, N, BLOCK_SIZE: tl.constexpr):",
            "    pid = tl.program_id(0)",
            "    temp_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    linear_offsets = temp_idx",
            "    linear_mask = linear_offsets < N",
        ]

        return "\n".join(
            kernel_def
            + body_code
            + [
                f"    tl.store(out_ptr + linear_offsets, {self.get_var_name(self.schedule[-1])}, mask=linear_mask)"
            ]
        )
