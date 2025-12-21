from dataclasses import dataclass
from typing import List, Literal, Optional

import triton
import triton.language as tl

from src.banhxeo.buffer import BinaryOps, LazyBuffer, LoadOps, UnaryOps


@dataclass
class InputArgument:
    buf: LazyBuffer
    type: Optional[Literal["ptr", "const"]] = None
    # shape, stride
    metadata: Optional[List[str]] = None


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
            if isinstance(buf.op, LoadOps):
                name = f"in_{len(self.input_args)}"
                if buf.op == LoadOps.FROM_CPU:  # assume load from CPU is linearly
                    self.input_args.append(InputArgument(buf, "ptr"))
                elif buf.op == LoadOps.CONST:
                    self.input_args.append(InputArgument(buf, "const"))
                elif buf.op == LoadOps.VIEW:
                    self.input_args.append(
                        InputArgument(buf, None, ["shape", "stride"])
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
        if len(children) == 1 and children[0].op == LoadOps.VIEW:
            return False

        return True

    def generate(self):
        # define the body and identify inputs
        body_code = []
        for buf in self.schedule:
            name = self.get_var_name(buf)

            if isinstance(buf.op, LoadOps):
                if self.should_materialize(buf):
                    if buf.op == LoadOps.CONST:
                        # hardcode value
                        body_code.append(f"    {name} = {name}_const")
                    elif buf.op == LoadOps.FROM_CPU:
                        body_code.append(
                            f"    {name} = tl.load({name}_ptr + linear_offsets, mask=linear_mask)"
                        )
                    elif buf.op == LoadOps.VIEW:
                        body_code.append(self.render_indexing(buf))
            elif isinstance(buf.op, BinaryOps):
                src0 = self.get_var_name(buf.src[0])
                src1 = self.get_var_name(buf.src[1])
                op_map = {
                    BinaryOps.ADD: "+",
                    BinaryOps.SUB: "-",
                    BinaryOps.MUL: "*",
                }
                body_code.append(f"    {name} = {src0} {op_map[buf.op]} {src1}")
            elif isinstance(buf.op, UnaryOps):
                src0 = self.get_var_name(buf.src[0])
                op_map = {
                    UnaryOps.LOG2: "tl.log2",
                    UnaryOps.EXP2: "tl.exp2",
                    UnaryOps.SIN: "tl.sin",
                    UnaryOps.SQRT: "tl.sqrt",
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
