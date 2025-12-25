## Current Features

- [x] Basic ops: `+, -, *, log, exp, sin, sqrt`
- [x] Lazy evaluation with computation graphs
- [x] View operations: `permute, slice, expand` (zero-copy!)
- [x] Broadcasting (naive but works)
- [x] Triton codegen for GPU execution and Pytorch for CPU execution (correctness validation only, don't use this backend)

## Roadmap

### v0.2: First Matmul
- [x] Solve the Reshape Boss: Implement the contiguous() check and the reshape logic.
- [x] Naive Matmul: Don't try to build a cuBLAS competitor. Write a simple Triton kernel that just works. (Block size 32, no fancy pipe-lining).
- [x] The MLP Forward Pass: Manually construct the weights for a small linear layer and run x @ W + b. Verify the numbers against PyTorch.

### v0.3: Autograd Engine

- Backward Ops
    - [x] Unary Op
    - [ ] Binary Op (partial)
    - [ ] Movement Op (partial)
    - [x] Ternary Op
- [ ] Reduce Ops: sum and max.
- [ ] Properly Broadcasting: need this for bias addition.
- [ ] Creation method (LoadOp.RAND)

### v0.4 - More Ops
- [ ] nn.Linear: Wrap Matmul + Add in a class.
- [ ] ReLU: Simple elementwise.
- [ ] LogSoftmax: Stability is key here.

### v0.5 - The Dream

- [ ] Data Loading: Just load the numpy arrays for MNIST (Maybe use Pytorch DataLoader).
- [ ] The CNN: Implement Conv2d. *Note*: Use im2col (unfold) to turn convolution into a MatMul. It's slower than custom Conv2D (or WinoGrad algorithm) but reuses Matmul kernel.
- [ ] Training: Run the loop.