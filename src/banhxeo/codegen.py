from typing import List

import triton
import triton.language as tl

from src.banhxeo.buffer import BinaryOps, LazyBuffer, LoadOps, UnaryOps


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
                if buf.op == LoadOps.FROM_CPU:
                    self.input_args.append((buf, "ptr"))
                elif buf.op == LoadOps.CONST:
                    self.input_args.append((buf, "const"))
                self.var_names[buf] = name
            else:
                self.var_names[buf] = f"temp_{len(self.var_names)}"
        return self.var_names[buf]

    def generate(self):
        # define the body and identify inputs
        body_code = []
        for buf in self.schedule:
            name = self.get_var_name(buf)

            if isinstance(buf.op, LoadOps):
                if buf.op == LoadOps.CONST:
                    # hardcode value
                    body_code.append(f"    {name} = {name}_const")
                elif buf.op == LoadOps.FROM_CPU:
                    body_code.append(
                        f"    {name} = tl.load({name}_ptr + offsets, mask=mask)"
                    )
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

        args_sig = [f"{self.get_var_name(buf)}_{sig}" for buf, sig in self.input_args]
        kernel_def = [
            "@triton.jit",
            f"def generated_kernel({', '.join(args_sig)}, out_ptr, N, BLOCK_SIZE: tl.constexpr):",
            "    pid = tl.program_id(0)",
            "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offsets < N",
        ]

        return "\n".join(
            kernel_def
            + body_code
            + [
                f"    tl.store(out_ptr + offsets, {self.get_var_name(self.schedule[-1])}, mask=mask)"
            ]
        )
