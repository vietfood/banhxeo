from dataclasses import dataclass
from typing import List, Literal, Optional

from banhxeo.core.buffer import BinaryOp, LazyBuffer, LoadOp, UnaryOp


@dataclass
class InputArgument:
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
        self.input_args = dict()
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
                    self.input_args[buf] = InputArgument("ptr")
                elif buf.op == LoadOp.CONST:
                    self.input_args[buf] = InputArgument("const")
                elif buf.op == LoadOp.VIEW or buf.op == LoadOp.CONTIGUOUS:
                    self.input_args[buf] = InputArgument(None, ["shape", "stride"])
                elif buf.op == LoadOp.FROM_NUMPY:
                    self.input_args[buf] = InputArgument("ptr", ["shape", "stride"])
                self.var_names[buf] = name
            else:
                self.var_names[buf] = f"temp_{len(self.var_names)}"
        return self.var_names[buf]

    def render_indexing(self, buf, only_compute_offset=False):
        src = buf.src[0]

        # If the source is a CONST, we don't care about strides/offsets.
        # Just skip it
        if src == LoadOp.CONST:
            return ""

        name = self.get_var_name(buf)
        src_name = self.get_var_name(src)
        code = [
            f"    {name}_offset = {buf.view.offset}",
            f"    {name}_idx = temp_idx",
        ]
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

        if only_compute_offset:
            return chr(10).join(code), f"{name}_offset"
        else:
            src_arg = self.input_args.get(src)
            sig = "" if src_arg is None else f"_{src_arg.type}"
            return chr(10).join(
                [
                    *code,
                    f"    {name} = tl.load({src_name}{sig} + {name}_offset)",
                ]
            ), name

    def should_materialize(self, buf):
        # Count children (buffers that use this as src)
        children = [b for b in self.schedule if buf in b.src]

        # Only materialize if it has multiple children or non-VIEW children
        if len(children) == 1 and children[0].op == LoadOp.VIEW:
            return False

        return True

    def handle_op_view(self, buf, name, output_name, contiguous=False):
        body_code = []
        view_src = buf.src[0]
        view_src_name = self.get_var_name(view_src)
        if not isinstance(view_src.op, LoadOp):
            idx_code, offset_var = self.render_indexing(buf, only_compute_offset=True)
            body_code.append(idx_code)
            if contiguous:
                body_code.append(f"    {name} = {view_src_name}")
            # Update the final store to use this scattered offset
            output_offset_name = offset_var
            output_name = view_src_name
        else:
            # Standard path: View of a Pointer (Gather)
            code, _ = self.render_indexing(buf)
            body_code.append(code)
            if contiguous:
                body_code.append(f"    {name} = {self.get_var_name(buf)}")
            output_offset_name = "linear_offsets"
        return output_offset_name, body_code, output_name

    def generate(self):
        # define the body and identify inputs
        body_code = []
        output_offset_name = "linear_offsets"
        output_name = ""

        for buf in self.schedule:
            name = self.get_var_name(buf)

            if isinstance(buf.op, LoadOp):
                if self.should_materialize(buf):
                    buf_arg = self.input_args.get(buf)
                    buf_sig = "" if buf_arg is None else f"_{buf_arg.type}"
                    if buf.op == LoadOp.CONST:
                        # hardcode value
                        body_code.append(f"    {name} = {name}{buf_sig}")
                    elif buf.op == LoadOp.FROM_CPU:
                        body_code.append(
                            f"    {name} = tl.load({name}{buf_sig} + linear_offsets, mask=linear_mask)"
                        )
                    if buf.op == LoadOp.CONTIGUOUS:
                        if buf.src[0].op == LoadOp.VIEW:
                            output_offset_name, view_body_code, output_name = (
                                self.handle_op_view(
                                    buf.src[0], name, output_name, contiguous=True
                                )
                            )
                            body_code.extend(view_body_code)
                        else:
                            # Otherwise just reference the source directly
                            body_code.append(
                                f"    {name} = {self.get_var_name(buf.src[0])}"
                            )
                    elif buf.op == LoadOp.VIEW:
                        output_offset_name, view_body_code, output_name = (
                            self.handle_op_view(
                                buf, name, output_name, contiguous=False
                            )
                        )
                        body_code.extend(view_body_code)
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
        for buf, args in self.input_args.items():
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
                f"    tl.store(out_ptr + {output_offset_name}, {output_name}, mask=linear_mask)"
            ]
        )
