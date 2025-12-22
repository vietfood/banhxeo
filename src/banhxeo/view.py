from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class View:
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int = 0

    def is_contiguous(self) -> bool:
        if len(self.shape) == 0 or sum(self.shape) == 0:
            return False
        # we create another view with "standard stride"
        return self.strides == self.create(self.shape).strides

    @staticmethod
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

    def broadcast_to(self, target_shape: Tuple[int, ...]) -> "View":
        """
        * Broadcasts this view to a new target shape.
        * Example: self=(3,1), target=(3,4) -> shape=(3,4), strides=(1,0)
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

    def slice(self, args: Tuple[Tuple[int, ...], ...]) -> "View":
        """
        * We assume args passed to slice is a tuple of (start, end) ranges for each dimension, e.g., ((0, 2), (1, 3)).
        * Right now we don't take account of step size
        """
        if len(self.shape) != len(args):
            raise ValueError(
                f"Slice arguments {args} is not match with size of shape {self.shape}"
            )

        if not all([len(arg) <= 2 for arg in args]):
            raise ValueError(
                f"Each argument of slice {args} must have size between 2 and 1"
            )

        new_shape = []
        new_offset = self.offset

        for i, arg in enumerate(args):
            if len(arg) == 1:
                start, end = 0, arg[0]
            elif len(arg) == 2:
                start, end = arg[0], arg[1]
            else:
                start, end = 0, self.shape[i]

            if start < 0 or end > self.shape[i] or start > end:
                raise ValueError(f"Slice {start}:{end} out of bounds...")

            new_shape.append(end - start)
            new_offset += start * self.strides[i]

        return View(tuple(new_shape), self.strides, new_offset)

    def reshape(self, new_shape: Tuple[int, ...]) -> "View":
        """
        Note that we always assume we can reshape freely
        """

        if all(x >= 0 for x in new_shape):
            raise ValueError(f"shape can't contain negative numbers {new_shape}")

        if 0 in self.shape:
            if 0 not in new_shape:
                raise ValueError(f"cannot reshape 0 size to {new_shape}")
            return View.create(new_shape)

        # Total size must match
        if math.prod(self.shape) != math.prod(new_shape):
            raise ValueError(f"Shape size mismatch, can't reshape {self.shape=} -> {new_shape=}")

        # after the asserts, it's okay to check contiguous
        if self.contiguous: 
            return View.create(new_shape)

        old_shape, old_strides = self.shape, self.strides
        new_strides = []
        
        # Pointer to where we are in the old shape
        old_idx = 0
        
        for new_dim in new_shape:
            # If new dimension is 1, stride can be anything (usually 0 or old stride)
            if new_dim == 1:
                new_strides.append(0) 
                continue
            
            # We need to cover 'new_dim' elements using the current old dimensions
            covered = 1
            current_stride = None
            
            while covered < new_dim and old_idx < len(old_shape):
                d = old_shape[old_idx]
                s = old_strides[old_idx]
                
                # If we are starting a new merged block
                if current_stride is None:
                    current_stride = s
                # If we are continuing a merge, check continuity
                else:
                    # Previous stride must equal (current dim * current stride)
                    # Wait, simpler check: 
                    # Does the memory layout flow continuously?
                    prev_d = old_shape[old_idx-1]
                    prev_s = old_strides[old_idx-1]
                    if prev_s != d * s:
                        raise ValueError(f"Cannot reshape non-contiguous view {self.shape} to {new_shape}")

                covered *= d
                old_idx += 1
            
            if covered != new_dim:
                 raise ValueError("Shape mismatch or dimension fragmentation")
            
            new_strides.append(current_stride)

        return View(new_shape, tuple(new_strides), self.offset)
