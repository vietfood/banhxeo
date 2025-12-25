from typing import List

import torch

from banhxeo.core.buffer import BinaryOp, LazyBuffer, LoadOp, TernaryOp, UnaryOp
from banhxeo.utils.helpers import DEBUG


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
                    BinaryOp.CMPLT: lambda x, y: (x < y).float(),
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
            elif isinstance(buf.op, TernaryOp):
                assert buf.src[0].realized is not None
                assert buf.src[1].realized is not None
                assert buf.src[2].realized is not None

                op_map = {
                    TernaryOp.WHERE: torch.where,
                }

                # torch.where expects bool for condition, ensuring we cast if needed
                condition = buf.src[0].realized.data.bool()
                x = buf.src[1].realized.data
                y = buf.src[2].realized.data

                buf.realized.data = op_map[buf.op](condition, x, y)

            if DEBUG >= 2:
                print(buf.realized.data)
