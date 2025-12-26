from dataclasses import dataclass
from typing import List, Literal, Optional

from banhxeo.core.buffer import BinaryOp, LazyBuffer, LoadOp, TernaryOp, UnaryOp
from banhxeo.utils.helpers import DEBUG


@dataclass
class InputArgument:
    type: Optional[Literal["ptr", "const"]] = None
    # shape, stride
    metadata: Optional[List[str]] = None


class TritonCodegen:
    def __init__(self, schedule: List[LazyBuffer]):
        self.schedule = schedule

        # Map LazyBuffer to Variable name
        self.var_names = dict()
        # Keep track of which input pointers we need
        self.input_args = dict()
        # Keep track of realized input
        self.realized_names = dict()

        self.output_offset = "linear_offsets"
        self.read_offset = "linear_offsets"
        self.output_name = ""

        # the gen code
        self.code = []

    def get_var_name(self, buf: LazyBuffer) -> str:
        if buf.realized is None:
            if buf not in self.var_names:
                # If it's a LoadOp, it needs a variable name that corresponds to a kernel argument
                if isinstance(buf.op, LoadOp):
                    name = f"in_{len(self.input_args)}"
                    if buf.op == LoadOp.FROM_PYTHON or buf.op == LoadOp.FROM_NONE:
                        # assume LoadOp.FROM_CPU is linearly
                        # note that DEFAULT create an empty Tensor so we assume it is a pointer too
                        self.input_args[buf] = InputArgument("ptr")
                    elif buf.op == LoadOp.FROM_CONST:
                        self.input_args[buf] = InputArgument("const")
                    elif buf.op == LoadOp.VIEW or buf.op == LoadOp.CONTIGUOUS:
                        self.input_args[buf] = InputArgument(None, ["shape", "stride"])
                    elif buf.op in (LoadOp.FROM_NUMPY, LoadOp.FROM_TORCH):
                        self.input_args[buf] = InputArgument("ptr", ["shape", "stride"])
                    self.var_names[buf] = name
                else:
                    self.var_names[buf] = f"temp_{len(self.var_names)}"
            return self.var_names[buf]
        else:  # for realized buffer, we treat it as an input
            if buf not in self.realized_names:
                self.realized_names[buf] = f"in_{len(self.input_args)}"
                self.input_args[buf] = InputArgument("ptr")
            return self.realized_names[buf]

    def render_indexing(self, buf, only_compute_offset=False):
        src = buf.src[0]

        # If the source is a CONST, we don't care about strides/offsets.
        # Just skip it
        if src == LoadOp.FROM_CONST:
            return ""

        name = self.get_var_name(buf)
        src_name = self.get_var_name(src)
        code = [
            f"    {name}_offset = {buf.view.offset}",
            f"    {name}_idx = {self.read_offset}",
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

    def visit_LoadOp(self, buf: LazyBuffer, name: str):
        if DEBUG >= 2:
            print(f"Visit LoadOp {str(buf.op)=} with name {name}")

        # If this VIEW is the output, we force linear storage.
        # The indexing math is handled in generate() to map inputs.
        if buf == self.schedule[-1] and buf.op == LoadOp.VIEW:
            self.output_offset = "linear_offsets"
            self.output_name = self.get_var_name(buf.src[0])
            return

        def handle_op_view(b, contiguous=False):
            view_src = b.src[0]
            view_src_name = self.get_var_name(view_src)
            if not isinstance(view_src.op, LoadOp):
                idx_code, offset_var = self.render_indexing(b, only_compute_offset=True)
                self.code.append(idx_code)
                if contiguous:
                    self.code.append(f"    {name} = {view_src_name}")
                # Update the final store to use this scattered offset
                self.output_offset = offset_var
                self.output_name = view_src_name
            else:
                code, _ = self.render_indexing(b)
                self.code.append(code)
                if contiguous:
                    self.code.append(f"    {name} = {self.get_var_name(b)}")
                self.output_offset = "linear_offsets"

        def should_materialize():
            # Count children (buffers that use this as src)
            children = [b for b in self.schedule if buf in b.src]

            # Only materialize if LoadOp has multiple children or has non-VIEW children
            if len(children) == 1 and children[0].op == LoadOp.VIEW:
                return False

            return True

        if should_materialize():
            buf_arg = self.input_args.get(buf)
            buf_sig = "" if buf_arg is None else f"_{buf_arg.type}"
            if buf.op == LoadOp.FROM_CONST:
                self.code.append(f"    {name} = {name}{buf_sig}")
            elif buf.op == LoadOp.FROM_PYTHON:
                self.code.append(
                    f"    {name} = tl.load({name}{buf_sig} + {self.read_offset}, mask=linear_mask)"
                )
            elif buf.op in (LoadOp.FROM_NUMPY, LoadOp.FROM_TORCH):
                code = [
                    f"    {name}_offset = {buf.view.offset}",
                    f"    {name}_idx = {self.read_offset}",
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
                self.code.append(
                    chr(10).join(
                        [
                            *code,
                            f"    {name} = tl.load({name}{buf_sig} + {name}_offset)",
                        ]
                    ),
                )
            elif buf.op == LoadOp.CONTIGUOUS:
                if buf.src[0].op == LoadOp.VIEW and buf.src[0].realized is None:
                    handle_op_view(buf.src[0], contiguous=True)
                else:
                    handle_op_view(buf, contiguous=True)
            elif buf.op == LoadOp.VIEW:
                handle_op_view(buf, contiguous=False)

    def visit_BinaryOp(self, buf: LazyBuffer, name: str):
        if DEBUG >= 2:
            print(f"Visit Binary {str(buf.op)=} with name {name}")

        src0 = self.get_var_name(buf.src[0])
        src1 = self.get_var_name(buf.src[1])

        if buf.op == BinaryOp.MAX:
            self.code.append(f"    {name} = tl.maximum({src0}, {src1})")
        else:
            op_map = {
                BinaryOp.ADD: "+",
                BinaryOp.SUB: "-",
                BinaryOp.MUL: "*",
                BinaryOp.CMPLT: "<",
                BinaryOp.DIV: "/",
                BinaryOp.MOD: "%",
            }
            self.code.append(f"    {name} = {src0} {op_map[buf.op]} {src1}")  # pyright: ignore[reportArgumentType]

    def visit_UnaryOp(self, buf: LazyBuffer, name: str):
        if DEBUG >= 2:
            print(f"Visit Unary {str(buf.op)=} with name {name}")

        src0 = self.get_var_name(buf.src[0])

        if buf.op == UnaryOp.CAST:
            self.code.append(f"    {name} = {src0}.to(tl.{buf.args[0].name})")
            return

        op_map = {
            UnaryOp.LOG: "tl.log({src})",
            UnaryOp.EXP: "tl.exp({src})",
            UnaryOp.SIN: "tl.sin({src})",
            UnaryOp.SQRT: "tl.sqrt({src})",
            UnaryOp.NEG: "-{src}",
        }
        self.code.append(f"    {name} = {op_map[op].format(src=src0)}")  # pyright: ignore[reportUndefinedVariable, reportArgumentType]

    def visit_Input(self, buf: LazyBuffer, name: str):
        if DEBUG >= 2:
            print(f"Visit Input {str(buf.op)=} with name {name}")

        buf_arg = self.input_args.get(buf)
        sig = "" if buf_arg is None else f"_{buf_arg.type}"
        self.code.append(
            f"    {name} = tl.load({name}{sig} + {self.read_offset}, mask=linear_mask)"
        )

    def visit_TernaryOp(self, buf: LazyBuffer, name: str):
        if DEBUG >= 2:
            print(f"Visit Ternary {str(buf.op)=} with name {name}")

        src0 = self.get_var_name(buf.src[0])  # Condition
        src1 = self.get_var_name(buf.src[1])  # True path
        src2 = self.get_var_name(buf.src[2])  # False path

        if buf.op == TernaryOp.WHERE:
            # tl.where(condition, true_val, false_val)
            self.code.append(f"    {name} = tl.where({src0}, {src1}, {src2})")

    def visit(self, buf: LazyBuffer, name: str):
        if buf.realized is not None:
            self.visit_Input(buf, name)
        else:
            if isinstance(buf.op, LoadOp):
                self.visit_LoadOp(buf, name)
            elif isinstance(buf.op, BinaryOp):
                self.visit_BinaryOp(buf, name)
            elif isinstance(buf.op, UnaryOp):
                self.visit_UnaryOp(buf, name)
            elif isinstance(buf.op, TernaryOp):
                self.visit_TernaryOp(buf, name)

    def generate(self):
        last_buf = self.schedule[-1]
        if last_buf.op == LoadOp.VIEW:
            # Generate the indexing code based on the Output View's strides
            # This calculates the offset into the Source data.
            code, name = self.render_indexing(last_buf, only_compute_offset=True)
            self.code.append(code)
            # All inputs should be read using this calculated offset
            self.read_offset = name

        for buf in self.schedule:
            self.visit(buf, self.get_var_name(buf))

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
            "@triton.heuristics(values={'BLOCK_SIZE': lambda args: min(triton.next_power_of_2(args['N']), 1024))",  # use Triton heuristics for better performance
            "@triton.jit",
            f"def generated_kernel({', '.join(args_sig)}, out_ptr, N, BLOCK_SIZE: tl.constexpr):",
            "    pid = tl.program_id(0)",
            "    temp_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    linear_offsets = temp_idx",
            "    linear_mask = linear_offsets < N",
        ]

        if self.output_name == "":
            self.output_name = self.get_var_name(self.schedule[-1])

        return "\n".join(
            kernel_def
            + self.code
            + [
                f"    tl.store(out_ptr + {self.output_offset}, {self.output_name}, mask=linear_mask)"
            ]
        )
