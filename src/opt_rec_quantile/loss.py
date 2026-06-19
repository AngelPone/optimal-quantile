import torch
import cvxpy as cp
import warnings


def approx_pinball_loss(x: torch.Tensor, beta: float, alpha: float):
    return torch.mean(torch.nn.functional.softplus(-x, beta=beta) + alpha * x)


def pinball_loss(x: torch.Tensor, alpha: float):
    return torch.mean(torch.where(x < 0, -(1 - alpha) * x, alpha * x))


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
    def forward(ctx, S, G, y, alpha, beta):
        with torch.no_grad():
            samples = torch.einsum("tnj,mn->tmj", y, S @ G)
            z = solve_approx_pinball_loss(samples, alpha, beta)
            ctx.alpha = alpha
            ctx.beta = beta
            ctx.save_for_backward(z, S, G, y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, S, G, y = ctx.saved_tensors
        beta = ctx.beta
        rf = torch.einsum("tnj,mn->tmj", y, S @ G)
        loss = beta * (rf - z[:, :, None])
        divide = torch.sigmoid(loss) * torch.sigmoid(-loss)
        left = divide.mean(dim=2)
        right = torch.einsum("km,tnj->tkmnj", S, y)
        right = torch.einsum("tnj,tnabj->tnabj", divide, right)
        right = right.mean(dim=(4))
        grad = right / left[:, :, None, None]
        grad = torch.einsum("tk,tkab->ab", grad_output, grad)
        return None, grad, None, None, None


if __name__ == "__main__":
    S = torch.Tensor([[1, 1], [0.0, 1.0]])
    G = torch.rand((2, 3), requires_grad=True)
    y = torch.rand((5, 3, 100))
    rf = torch.einsum("tnj,kn->tkj", y, S @ G)

    q = ApproxPinballLoss.apply(S, G, y, 0.95, 20)
    print(
        f"estimated quantile: {q.detach().numpy()}, true quantile: {torch.quantile(rf, 0.95, dim=2).detach().numpy()}"
    )
    q.mean().backward()
    print(G.grad)
