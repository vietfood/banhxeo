import torch as t


def main():
    print("Hello from banhxeo!")


if __name__ == "__main__":
    A = t.tensor([1, 2, 3, 4], dtype=t.float32)
    B = t.tensor([5, 6, 7, 8], dtype=t.float32)
    print(A)
    print(B)
    print(A.dot(B))
