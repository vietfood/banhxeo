from typing import Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: Optional[int] = None  # might be useful for all models

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.model_dump().items())
        return f"{self.__class__.__name__}({params})"


class GenerateConfig(BaseModel):
    sampling: str = "greedy"
    max_length: Optional[int] = None  # generate until sep/end token
    k: Optional[int] = None  # for top K
    p: Optional[float] = None  # for top P
    temp: Optional[int] = None  # for temperature
