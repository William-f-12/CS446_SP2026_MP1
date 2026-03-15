import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mini_torch.tensor import Tensor
from mini_torch.ops import Add, Mul, Neg, MatMul, Mean, ReLU, Pow, Sigmoid

EPS = 1e-3
ATOL = 2e-2
RTOL = 2e-2


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _np_allclose(a, b, atol=ATOL, rtol=RTOL):
    return np.allclose(a, b, atol=atol, rtol=rtol)


def _finite_diff_grad(f, x, eps=EPS):
    grad = np.zeros_like(x, dtype=np.float32)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x[idx])
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def _check_grad_unary(op_apply, f_np, x0, name):
    x_np = x0.astype(np.float32)
    x = Tensor(x_np.copy(), requires_grad=True)

    y = op_apply(x)
    assert _np_allclose(y.detach().numpy(), f_np(x_np)), f"[{name}] forward mismatch"

    loss = y.mean()
    x.zero_grad()
    loss.backward()

    def f_scalar(x_arr):
        return float(np.array(f_np(x_arr), dtype=np.float32).mean())

    grad_num = _finite_diff_grad(f_scalar, x_np.copy())
    assert _np_allclose(x.grad, grad_num), f"[{name}] backward mismatch"


def _check_grad_binary(op_apply, f_np, a0, b0, name):
    a_np = a0.astype(np.float32)
    b_np = b0.astype(np.float32)

    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)

    y = op_apply(a, b)
    assert _np_allclose(y.detach().numpy(), f_np(a_np, b_np)), f"[{name}] forward mismatch"

    loss = y.mean()
    a.zero_grad()
    b.zero_grad()
    loss.backward()

    def f_scalar_a(a_arr):
        return float(np.array(f_np(a_arr, b_np), dtype=np.float32).mean())

    def f_scalar_b(b_arr):
        return float(np.array(f_np(a_np, b_arr), dtype=np.float32).mean())

    ga_num = _finite_diff_grad(f_scalar_a, a_np.copy())
    gb_num = _finite_diff_grad(f_scalar_b, b_np.copy())

    assert _np_allclose(a.grad, ga_num), f"[{name}] backward mismatch (a)"
    assert _np_allclose(b.grad, gb_num), f"[{name}] backward mismatch (b)"


# =================================================
# ADD
# =================================================

@pytest.mark.op("add")
def test_add_basic():
    # Case: basic (same shape, 2D)
    a0 = np.random.randn(3,4)
    b0 = np.random.randn(3,4)
    _check_grad_binary(lambda a,b: Add.apply(a,b), lambda a,b: a+b, a0, b0, "Add-basic")


@pytest.mark.op("add")
def test_add_highdim():
    # Case: high-dimensional tensors (3D)
    a0 = np.random.randn(2,3,4)
    b0 = np.random.randn(2,3,4)
    _check_grad_binary(lambda a,b: Add.apply(a,b), lambda a,b: a+b, a0, b0, "Add-highdim")


@pytest.mark.op("add")
def test_add_broadcast():
    # Case: broadcast (last dimension broadcasted)
    a0 = np.random.randn(3,4)
    b0 = np.random.randn(4)
    _check_grad_binary(lambda a,b: Add.apply(a,b), lambda a,b: a+b, a0, b0, "Add-broadcast")


@pytest.mark.op("add")
def test_add_constant():
    # Case: constant operand (scalar constant)
    x0 = np.random.randn(3, 4).astype(np.float32)
    x = Tensor(x0.copy(), requires_grad=True)

    y = Add.apply(x, 2.0)
    y.mean().backward()

    assert x.grad is not None
    ref = np.ones_like(x0, dtype=np.float32) / x0.size
    np.testing.assert_allclose(x.grad, ref, atol=1e-6, rtol=1e-6)


# =================================================
# MUL
# =================================================

@pytest.mark.op("mul")
def test_mul_basic():
    # Case: basic (same shape, 2D)
    a0 = np.random.randn(3,4)
    b0 = np.random.randn(3,4)
    _check_grad_binary(lambda a,b: Mul.apply(a,b), lambda a,b: a*b, a0, b0, "Mul-basic")


@pytest.mark.op("mul")
def test_mul_highdim():
    # Case: high-dimensional tensors (3D)
    a0 = np.random.randn(2,3,4)
    b0 = np.random.randn(2,3,4)
    _check_grad_binary(lambda a,b: Mul.apply(a,b), lambda a,b: a*b, a0, b0, "Mul-highdim")


@pytest.mark.op("mul")
def test_mul_broadcast():
    # Case: broadcast (vector broadcasted over matrix)
    a0 = np.random.randn(3,4)
    b0 = np.random.randn(4)
    _check_grad_binary(lambda a,b: Mul.apply(a,b), lambda a,b: a*b, a0, b0, "Mul-broadcast")

@pytest.mark.op("mul")
def test_mul_constant():
    # Case: constant operand (scalar multiplier)
    x0 = np.random.randn(3,4)
    x = Tensor(x0.copy(), requires_grad=True)

    y = Mul.apply(x, 3.0)
    y.mean().backward()

    ref = np.ones_like(x0) * 3.0 / x0.size
    assert _np_allclose(x.grad, ref)
# =================================================
# NEG
# =================================================

@pytest.mark.op("neg")
def test_neg_basic():
    # Case: basic (2D)
    x0 = np.random.randn(3,4)
    _check_grad_unary(lambda x: Neg.apply(x), lambda x: -x, x0, "Neg-basic")


@pytest.mark.op("neg")
def test_neg_highdim():
    # Case: high-dimensional tensor (3D)
    x0 = np.random.randn(2,3,4)
    _check_grad_unary(lambda x: Neg.apply(x), lambda x: -x, x0, "Neg-highdim")


# =================================================
# MEAN (basic only)
# =================================================

@pytest.mark.op("mean")
def test_mean_basic():
    # Case: basic full-tensor mean reduction
    x0 = np.random.randn(3,4,5)
    x = Tensor(x0.copy(), requires_grad=True)

    y = Mean.apply(x)
    assert _np_allclose(y.detach().numpy(), np.array(x0.mean(), dtype=np.float32))

    x.zero_grad()
    y.backward()
    ref = np.ones_like(x0) / float(x0.size)
    assert _np_allclose(x.grad, ref)


# =================================================
# RELU
# =================================================

@pytest.mark.op("relu")
def test_relu_basic():
    # Case: basic (2D element-wise)
    f = lambda x: x * (x>0)
    x0 = np.random.randn(3,4)
    _check_grad_unary(lambda x: ReLU.apply(x), f, x0, "ReLU-basic")


@pytest.mark.op("relu")
def test_relu_highdim():
    # Case: high-dimensional tensor (3D element-wise)
    f = lambda x: x * (x>0)
    x0 = np.random.randn(2,3,4)
    _check_grad_unary(lambda x: ReLU.apply(x), f, x0, "ReLU-highdim")


# =================================================
# POW
# =================================================

@pytest.mark.op("pow")
def test_pow_basic():
    # Case: basic element-wise power (2D)
    a0 = np.random.rand(3,4)+0.5
    b0 = np.full((3,4), 2.0)
    _check_grad_binary(lambda a,b: Pow.apply(a,b), np.power, a0, b0, "Pow-basic")


@pytest.mark.op("pow")
def test_pow_highdim():
    # Case: high-dimensional element-wise power (3D)
    a0 = np.random.rand(2,3,4)+0.5
    b0 = np.full((2,3,4), 2.0)
    _check_grad_binary(lambda a,b: Pow.apply(a,b), np.power, a0, b0, "Pow-highdim")
    
@pytest.mark.op("pow")
def test_pow_broadcast():
    # Case: broadcast exponent (vector over matrix)
    a0 = np.random.rand(3,4)+0.5
    b0 = np.random.rand(4)+0.5
    _check_grad_binary(
        lambda a,b: Pow.apply(a,b),
        np.power,
        a0,
        b0,
        "Pow-broadcast",
    )
    
@pytest.mark.op("pow")
def test_pow_constant_exp():
    # Case: constant exponent (scalar)
    x0 = np.random.rand(3,4)+0.5
    x = Tensor(x0.copy(), requires_grad=True)

    y = Pow.apply(x, 2.0)
    y.mean().backward()

    ref = 2.0 * x0 / x0.size
    assert _np_allclose(x.grad, ref)


# =================================================
# SIGMOID
# =================================================

@pytest.mark.op("sigmoid")
def test_sigmoid_basic():
    # Case: basic (2D element-wise)
    f = lambda x: 1/(1+np.exp(-x))
    x0 = np.random.randn(3,4)
    _check_grad_unary(lambda x: Sigmoid.apply(x), f, x0, "Sigmoid-basic")


@pytest.mark.op("sigmoid")
def test_sigmoid_highdim():
    # Case: high-dimensional tensor (3D element-wise)
    f = lambda x: 1/(1+np.exp(-x))
    x0 = np.random.randn(2,3,4)
    _check_grad_unary(lambda x: Sigmoid.apply(x), f, x0, "Sigmoid-highdim")


# =================================================
# MATMUL
# =================================================

@pytest.mark.op("matmul")
def test_matmul_matrix():
    # Case: matrix @ matrix (2D x 2D)
    a0 = np.random.randn(3,4)
    b0 = np.random.randn(4,2)
    _check_grad_binary(lambda a,b: MatMul.apply(a,b), lambda a,b: a@b, a0, b0, "MatMul-matrix")


@pytest.mark.op("matmul")
def test_matmul_matvec():
    # Case: matrix @ vector (2D x 1D)
    a0 = np.random.randn(3,4)
    b0 = np.random.randn(4) 
    _check_grad_binary(lambda a,b: MatMul.apply(a,b), lambda a,b: a@b, a0, b0, "MatMul-matvec")


@pytest.mark.op("matmul")
def test_matmul_vecmat():
    # Case: vector @ matrix (1D x 2D)
    a0 = np.random.randn(4)
    b0 = np.random.randn(4,3)
    _check_grad_binary(lambda a,b: MatMul.apply(a,b), lambda a,b: a@b, a0, b0, "MatMul-vecmat")


@pytest.mark.op("matmul")
def test_matmul_vecvec():
    # Case: vector @ vector (1D x 1D, scalar output)
    a0 = np.random.randn(4)
    b0 = np.random.randn(4)
    _check_grad_binary(lambda a,b: MatMul.apply(a,b), lambda a,b: a@b, a0, b0, "MatMul-vecvec")


@pytest.mark.op("matmul")
def test_matmul_batch():
    # Case: batched matrix multiplication (3D x 3D)
    a0 = np.random.randn(2,3,4)
    b0 = np.random.randn(2,4,5)
    _check_grad_binary(lambda a,b: MatMul.apply(a,b), lambda a,b: a@b, a0, b0, "MatMul-batch")
    
@pytest.mark.op("matmul")
def test_matmul_broadcast_batch():
    # Case: batch broadcasting
    a0 = np.random.randn(2,3,4)
    b0 = np.random.randn(4,5)  # no batch dim
    _check_grad_binary(
        lambda a,b: MatMul.apply(a,b),
        lambda a,b: a @ b,
        a0,
        b0,
        "MatMul-broadcast-batch",
    )