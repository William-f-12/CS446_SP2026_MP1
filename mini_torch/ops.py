from typing import Any, Optional, Tuple

import numpy as np

from .tensor import Tensor, as_tensor

"""
Context object used to store information needed for backward computation.

It allows the forward() method to save intermediate tensors or values that
will be required later to compute gradients in backward().
"""


class Context:
    def __init__(self):
        self.saved_tensors: Tuple[np.ndarray, ...] = ()
        self.saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *xs: np.ndarray) -> None:
        self.saved_tensors = tuple(xs)

    def save_values(self, *vals: Any) -> None:
        self.saved_values = tuple(vals)


class Function:
    """
    Graph node: one instance per forward call.
    """

    def __init__(self, ctx: Context, parents: Tuple["Tensor", ...]):
        self.ctx = ctx
        self.parents = parents

    @staticmethod
    def forward(ctx: Context, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(
        ctx: Context, grad_out: np.ndarray
    ) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Any) -> "Tensor":
        """
        Apply this autograd Function to the given inputs.

        Inputs:
            *inputs:
                Positional inputs to the operation. Each input may be a Tensor,
                NumPy array, or Python scalar.

        Returns:
            Tensor:
                The output Tensor containing the forward result, with requires_grad
                set appropriately and grad_fn pointing to the creating Function
                when gradient tracking is enabled.

        Side Effects:
            - Sets parents and Context to connect the computation graph.
            - Executes the forward pass.
            - Attaches this Function instance to the output Tensor as grad_fn.
        """
        parents = tuple(as_tensor(x) for x in inputs)
        req = any(p.requires_grad for p in parents)
        # 1) Create a Context Object, run cls.forward(ctx, ...) to compute the output value.
        # The forward method in its subclass will compute the forward pass and store necessary information for backward in the Context Object.
        # Make sure to pass the raw data (np.ndarray) instead of Tensor to the forward method for numerical computation.
        ctx = Context()
        output = cls.forward(ctx, *(p.data for p in parents))

        # 2）Create the output Tensor (set data and requires_grad appropriately).
        output_tensor = Tensor(output, requires_grad=req)

        # 3) Create the computation-graph node appropriately and attach it to the output Tensor (.grad_fn).
        # Note cls(the first argument) is the subclass of Function, so you can create the node by cls(...), with appropriate parameters.
        if req:
            output_tensor.grad_fn = cls(ctx, parents)

        return output_tensor

    @staticmethod
    def reduce_shape(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad


# ===== Functions (ops) =====
class Add(Function):
    """
    Forward:
        Inputs:
            ctx (Context):
                Context object for saving information needed in backward().
            a (np.ndarray):
                First input array.
            b (np.ndarray):
                Second input array.

        Returns:
            np.ndarray:
                Element-wise sum of a and b.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx, a, b):
        # save the shape only since the gradients is always 1
        ctx.save_values(a.shape, b.shape)
        return a + b

    """
    Backward:
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b.
    """

    @staticmethod
    def backward(ctx, grad_out):
        a_shape, b_shape = ctx.saved_values

        # reshape grad_out to the shape of a and b
        return Function.reduce_shape(grad_out, a_shape), Function.reduce_shape(
            grad_out, b_shape
        )


class Pow(Function):
    """
    Forward:
        Inputs:
            ctx (Context):
                Context object for saving information needed in backward().
            a (np.ndarray):
                Base input array.
            b (np.ndarray):
                Exponent input array.

        Returns:
            np.ndarray:
                Element-wise power a ** b.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
              required to compute gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a**b

    """
    Backward:
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b.
    """

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        ga = grad_out * b * (a ** (b - 1))
        safe_a = np.where(a > 0, a, 1.0)
        gb = grad_out * (a**b) * np.log(safe_a)

        return Function.reduce_shape(ga, a.shape), Function.reduce_shape(gb, b.shape)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        """
        Inputs:
            ctx (Context):
                Context object for saving information needed in backward().
            a (np.ndarray):
                First input array.
            b (np.ndarray):
                Second input array.

        Returns:
            np.ndarray:
                Element-wise product of a and b. The output shape is the
                broadcasted shape of the inputs.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output, with the same shape as the forward output.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b. Each gradient has
                the same shape as its corresponding input.
        """
        a, b = ctx.saved_tensors
        return Function.reduce_shape(grad_out * b, a.shape), Function.reduce_shape(
            grad_out * a, b.shape
        )


class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Inputs:
            ctx (Context):
                Context object for backward computation.
            x (np.ndarray):
                Input array.

        Returns:
            np.ndarray:
                Element-wise negation of x, with the same shape as x.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        ctx.save_values(x.shape)
        return -x

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object from forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x, with the same shape as x.
        """
        (x_shape,) = ctx.saved_values
        return Function.reduce_shape(-grad_out, x_shape)


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        """
        Inputs:
            ctx (Context):
                Context object for saving backward information.
            a (np.ndarray):
                Left matrix operand.
            b (np.ndarray):
                Right matrix operand.

        Returns:
            np.ndarray:
                Matrix product of a and b.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        a_shape, b_shape = a.shape, b.shape

        # adjust 1d vector to 2d matrix for matmul
        a_was_1d = a.ndim == 1
        b_was_1d = b.ndim == 1
        a2 = a[None, :] if a_was_1d else a  # (n,) -> (1,n)
        b2 = b[:, None] if b_was_1d else b  # (n,) -> (n,1)

        out = a2 @ b2

        # squeeze the output back to the correct shape
        if a_was_1d and b_was_1d:  # (n,)@(n,) -> ()
            out = out.squeeze(-1).squeeze(-1)
        elif a_was_1d:  # (n,)@(...,n,k) -> (...,k)
            out = np.squeeze(out, axis=-2)
        elif b_was_1d:  # (...,m,n)@(n,) -> (...,m)
            out = np.squeeze(out, axis=-1)

        ctx.save_for_backward(a2.astype(np.float32), b2.astype(np.float32))
        ctx.save_values(a_shape, b_shape, a_was_1d, b_was_1d)
        return np.array(out, dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b.

        Note:
            You might find np.swapaxes useful here.
        """
        a2, b2 = ctx.saved_tensors
        a_shape, b_shape, a_was_1d, b_was_1d = ctx.saved_values
        g = np.array(grad_out, dtype=np.float32)

        if a_was_1d and b_was_1d:  # () -> (1,1)
            g = g.reshape(1, 1)
        elif a_was_1d:  # (...,k) -> (...,1,k)
            g = np.expand_dims(g, axis=-2)
        elif b_was_1d:  # (...,m) -> (...,m,1)
            g = np.expand_dims(g, axis=-1)

        # gradients of matmul: C = A@B, dA = dC @ B^T, dB = A^T @ dC
        ga = g @ np.swapaxes(b2, -1, -2)
        gb = np.swapaxes(a2, -1, -2) @ g

        # squeeze the gradients back to the correct shape
        if a_was_1d:
            ga = np.squeeze(ga, axis=-2)
        if b_was_1d:
            gb = np.squeeze(gb, axis=-1)

        ga = Function.reduce_shape(ga, a_shape).astype(np.float32)
        gb = Function.reduce_shape(gb, b_shape).astype(np.float32)
        return ga, gb


# Example ops.
class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_values(x.shape)
        return np.array(x.sum(), dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        (sh,) = ctx.saved_values
        return (np.ones(sh, dtype=np.float32) * grad_out,)


class Mean(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Inputs:
            ctx (Context):
                Context object for backward computation.
            x (np.ndarray):
                Input array.

        Returns:
            np.ndarray:
                A scalar array containing the mean of all elements in x.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        ctx.save_values(x.shape)
        return np.array(x.mean(), dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the scalar output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x, with the same shape as x.
        """
        (x_shape,) = ctx.saved_values
        x_size = np.prod(x_shape)
        return (np.ones(x_shape, dtype=np.float32) * grad_out / x_size,)


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Inputs:
            ctx (Context):
                Context object for backward computation.
            x (np.ndarray):
                Input array.

        Returns:
            np.ndarray:
                Output array where each element is max(x, 0).

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        mask = x > 0
        ctx.save_for_backward(mask)
        return np.maximum(x, 0).astype(np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x.
        """
        (mask,) = ctx.saved_tensors
        gx = np.array(grad_out, dtype=np.float32) * mask.astype(
            np.float32
        )  # should have same shape
        return (gx,)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.empty_like(x, dtype=np.float32)

        pos_mask = x >= 0
        neg_mask = ~pos_mask

        out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

        exp_x = np.exp(x[neg_mask])
        out[neg_mask] = exp_x / (1.0 + exp_x)

        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x.
        """
        (out,) = ctx.saved_tensors
        gx = grad_out * out * (1 - out)
        return (gx.astype(np.float32),)


class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, target):
        """
        logits: (N, C)
        target:
            - (N,)   class indices
            - (N, C) class probabilities (including one-hot)
        returns:
            scalar mean cross-entropy
        """
        if logits.ndim != 2:
            raise ValueError("CrossEntropy expects logits with shape (N, C).")

        N, C = logits.shape

        # stable log_softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        probs = exp_shifted / sum_exp
        log_probs = shifted - np.log(sum_exp)

        # Case 1: target is class indices, shape (N,)
        if target.ndim == 1:
            if target.shape[0] != N:
                raise ValueError("Target with class indices must have shape (N,).")

            target_idx = target.astype(np.int64)
            if np.any(target_idx < 0) or np.any(target_idx >= C):
                raise ValueError("Target contains invalid class index.")

            loss = -np.mean(log_probs[np.arange(N), target_idx]).astype(np.float32)
            target_dist = np.zeros((N, C), dtype=np.float32)
            target_dist[np.arange(N), target_idx] = 1.0

        # Case 2: target is class probabilities, shape (N, C)
        elif target.ndim == 2:
            if target.shape != (N, C):
                raise ValueError("Target probabilities must have shape (N, C).")

            target_dist = target.astype(np.float32)
            loss = -np.mean(np.sum(target_dist * log_probs, axis=1)).astype(np.float32)

        else:
            raise ValueError("Target must have shape (N,) or (N, C).")

        ctx.save_for_backward(probs.astype(np.float32), target_dist.astype(np.float32))
        ctx.save_values(N)
        return np.array(loss, dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        probs, target_dist = ctx.saved_tensors
        (N,) = ctx.saved_values

        grad_logits = (probs - target_dist) / float(N)
        grad_logits = (grad_logits * grad_out).astype(np.float32)
        return grad_logits, None
