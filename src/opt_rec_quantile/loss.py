import torch
import cvxpy as cp
import warnings


def approx_pinball_loss(
    x: torch.Tensor, beta: float, alpha: float, weights: torch.Tensor | None = None
):
    if weights is None:
        return torch.mean(torch.nn.functional.softplus(-x, beta=beta) + alpha * x)
    else:
        loss = torch.nn.functional.softplus(-x, beta=beta) + alpha * x
        loss = loss.mean(dim=0) / weights
        return loss.mean()


def pinball_loss(x: torch.Tensor, alpha: float, weights: torch.Tensor | None = None):
    if weights is None:
        return torch.mean(torch.where(x < 0, -(1 - alpha) * x, alpha * x))
    else:
        loss = torch.where(x < 0, -(1 - alpha) * x, alpha * x)
        loss = loss.mean(dim=0) / weights
        return loss.mean()


def solve_approx_pinball_loss(
    r: torch.Tensor, alpha: float, beta: float, solver="Empirical"
):
    def empirical_quantile():
        idx = int(torch.ceil(torch.as_tensor(r.shape[2] * alpha)).item()) - 1
        idx = max(0, min(idx, r.shape[2] - 1))
        return torch.sort(r, dim=-1)[0][:, :, idx]

    if solver != "Empirical":
        qs = []
        for t in range(r.shape[0]):
            window = []
            for i in range(r.shape[1]):
                smps = r[t, i, :]
                z = cp.Variable(1)
                d = smps - z
                inner = cp.logistic(-beta * d) / beta + alpha * d
                obj = cp.sum(inner)
                problem = cp.Problem(cp.Minimize(obj))
                try:
                    problem.solve(solver=solver)
                    warnings.warn(f"{solver} failed, using empirical quantile")
                    window.append(torch.tensor(z.value))
                except:
                    window.append(
                        torch.quantile(
                            r, torch.as_tensor(alpha, dtype=r.dtype)
                        ).reshape((-1))
                    )
            window = torch.stack(window)
            qs.append(window)
        return torch.stack(qs)
    else:
        return empirical_quantile()


class ApproxPinballLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, G, rf, y, alpha, beta):
        with torch.no_grad():
            z = solve_approx_pinball_loss(rf, alpha, beta)
            ctx.beta = beta
            ctx.save_for_backward(z, S, rf, y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, S, rf, y = ctx.saved_tensors
        m, n = S.shape
        beta = ctx.beta
        loss = beta * (rf - z[:, :, None])
        p = torch.sigmoid(loss)
        divide = p * (1 - p)
        left = divide.mean(dim=2)

        def logic1():
            right = torch.einsum("km,tnj->tkmnj", S, y)
            right = torch.einsum("tnj,tnabj->tnabj", divide, right)
            right = right.mean(dim=(4))
            grad = right / left[:, :, None, None]
            grad = torch.einsum("tk,tkab->ab", grad_output, grad)
            return grad

        def logic2():
            weight = grad_output / left
            core = (
                torch.einsum(
                    "tk,tkj,tnj->kn",
                    weight,
                    divide,
                    y,
                )
                / y.shape[-1]
            )
            grad = S.T @ core
            return grad

        def logic3():
            weight = grad_output / left
            grad = (
                torch.einsum(
                    "km,tk,tkj,tnj->mn",
                    S,
                    weight,
                    divide,
                    y,
                )
                / y.shape[-1]
            )
            return grad

        grad = logic2() if m > n // 2 else logic3()

        return None, grad, None, None, None, None


if __name__ == "__main__":
    S = torch.Tensor([[1, 1], [0.0, 1.0]])
    G = torch.rand((2, 3), requires_grad=True)
    y = torch.rand((5, 3, 100))

    q = ApproxPinballLoss.apply(S, G, y, 0.95, 20)

    q.mean().backward()
    print(G.grad)
