from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name: str

    def __repr__(self):
        return f"dtypes.{self.name}"


class dtypes:
    float32 = DType("float32")
    int32 = DType("int32")
    bool = DType("bool")
