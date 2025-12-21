import functools
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class View:
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int = 0

    def contiguous(self):
        # force the view to be contiguous
        if not self.is_contiguous():
            self.strides = self.create(self.shape).strides

    def is_contiguous(self) -> bool:
        if len(self.shape) == 0 or sum(self.shape) == 0:
            return False
        # we create another view with "standard stride"
        return self.strides == self.create(self.shape).strides

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def create(
        shape, stride: Optional[Tuple[int, ...]] = None, offset: Optional[int] = None
    ) -> "View":
        def create_stride():
            # Helper to create a contiguous view for a new tensor
            # shape (3, 2) -> strides (2, 1)
            strides = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
            return tuple(strides)

        if stride is None:
            new_stride = create_stride()
        else:
            # TODO: Check the correctness of input stride
            new_stride = stride

        return View(shape, new_stride, 0 if offset is None else offset)

    def can_broadcast(self, target_shape: Tuple[int, ...]) -> bool:
        # Right-align shapes
        # self:   (   3, 2)
        # target: (5, 3, 2)
        dims_self = len(self.shape)
        dims_target = len(target_shape)

        for i in range(min(dims_self, dims_target)):
            dim_s = self.shape[dims_self - 1 - i]
            dim_t = target_shape[dims_target - 1 - i]
            if dim_s != dim_t and dim_s != 1 and dim_t != 1:
                return False
        return True

    @functools.lru_cache(maxsize=None)
    def broadcast_to(self, target_shape: Tuple[int, ...]) -> "View":
        """
        Broadcasts this view to a new target shape.
        Example: self=(3,1), target=(3,4) -> shape=(3,4), strides=(1,0)
        """
        assert self.can_broadcast(target_shape)

        # 1. Pad dimensions on the left
        # If self is (3, 2) and target is (5, 3, 2), treat self as (1, 3, 2)
        ndim_diff = len(target_shape) - len(self.shape)
        padded_shape = (1,) * ndim_diff + self.shape
        padded_strides = (
            0,
        ) * ndim_diff + self.strides  # 0 stride for new phantom dims

        new_strides = []

        # 2. Iterate and update strides
        for dim_s, dim_t, stride_s in zip(padded_shape, target_shape, padded_strides):
            if dim_s == dim_t:
                new_strides.append(stride_s)
            elif dim_s == 1:
                # We are expanding a 1 to N. Stride becomes 0!
                new_strides.append(0)
            else:
                raise ValueError(f"Impossible broadcast: {dim_s} -> {dim_t}")

        return View(target_shape, tuple(new_strides))

    @functools.lru_cache(maxsize=None)
    def compute_offset(self, indices: Tuple[int, ...]) -> int:
        """
        Return real offset in memory buffer of current indices
        """
        if len(self.shape) != len(indices):
            raise ValueError(f"Indices {indices} doesn't match with shape {self.shape}")

        result = self.offset
        for i in range(len(indices)):
            idx = indices[i]
            if indices[i] < 0:
                # for this, we would take index i "backward"
                # for example shape (2, 3) and i is (1, -1)
                # then real indices will be (1, 2)
                idx = self.shape[i] + indices[i]
            if idx >= self.shape[i] or idx < 0:
                raise ValueError(
                    f"Index {indices[i]} is out of bounds for axis {i} with size {self.shape[i]}"
                )
            result += idx * self.strides[i]

        return result

    @functools.lru_cache(maxsize=None)
    def permute(self, new_axis: Tuple[int, ...]) -> "View":
        if len(new_axis) != len(self.shape):
            raise ValueError(
                f"Permutation new axis {new_axis} doesn't match with shape {self.shape}"
            )

        if not all([ax < len(self.shape) for ax in new_axis]):
            raise ValueError(f"Invalid permute axis {new_axis} for shape {self.shape}")

        target_shape = [0] * len(self.shape)
        target_stride = [0] * len(self.shape)

        for i in range(len(self.shape)):
            target_shape[i] = self.shape[new_axis[i]]
            target_stride[i] = self.strides[new_axis[i]]

        return View(tuple(target_shape), tuple(target_stride))

    @functools.lru_cache(maxsize=None)
    def slice(self, args: Tuple[int, ...]) -> "View":
        if len(self.shape) != len(args):
            raise ValueError(
                f"Slice arguments {args} is not match with size of shape {self.shape}"
            )
        # TODO: Implement slice
