from banhxeo.tensor import Tensor

# create tensor
a = Tensor([1, 2, 3, 4])
b = Tensor([3, 4, 5, 6])
c = a + b
d = c * a
e = d.log()

# realized
print(e.realize())
