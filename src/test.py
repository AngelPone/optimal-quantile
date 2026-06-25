# %%

from simulation.config import data_catalog

alpha = 0.2
data_S1 = data_catalog[f"simulation_S1"].load()
base_S1 = data_catalog[f"simulation_base_S1"].load()
rf_S1 = data_catalog[f"simulation_rf_S1_{alpha}"].load()
samples_S1 = data_catalog[f"simulation_samples_S1_{alpha}"].load()

data_S4 = data_catalog[f"simulation_S4"].load()
base_S4 = data_catalog[f"simulation_base_S4"].load()
samples_S4 = data_catalog[f"simulation_samples_S4_{alpha}"].load()
rf_S4 = data_catalog[f"simulation_rf_S4_{alpha}"].load()

# %%
import torch


def pinball_loss(x: torch.Tensor, alpha: float):
    return torch.mean(torch.where(x < 0, -(1 - alpha) * x, alpha * x))


# %%
d4 = torch.as_tensor(data_S4[0][-50:, :])
{
    method: pinball_loss(d4 - torch.quantile(f, alpha, 2), alpha)
    for method, f in samples_S4.items()
}
# %%
d1 = torch.as_tensor(data_S1[0][-50:, :])
{
    method: pinball_loss(d1 - torch.quantile(f, alpha, dim=2), alpha)
    for method, f in samples_S1.items()
}

# %%
from utils import expanding_window

d4_o = expanding_window(1000, 850, 1, data_S4[0]).collect_test().squeeze()[-50:, :]

# %%
samples_S1["base"].shape
# %%

rf_S1["model"].pinball_loss_history_

rf_S1["model"].smooth_loss_history_

# %%
import matplotlib.pyplot as plt

# plt.plot(rf_S1["model"].pinball_loss_history_)
plt.plot(rf_S4["model"].pinball_loss_history_)
plt.plot(rf_S4["model"].smooth_loss_history_)


# %%
import torch

d = torch.distributions.Normal(0.0, 1.0)
torch.normal(d.loc.expand(100), d.scale.expand(100)).shape

# %%
S = rf_S1["model"].S
torch.linalg.solve(S.T @ S, S.T)

# %%
0.99**50
torch.randint(0, 10, (torch.ceil(torch.as_tensor(10 * 0.9)).item(),))

# %%
from tourism.config import data_catalog

q = 0.2
samples_normal = data_catalog[f"tourism_samples_{q}_normal"].load()
true_y = torch.tensor(data_catalog["tourism"].load()[-60:,])
basef = data_catalog["tourism_base"].load()

# %%
import matplotlib.pyplot as plt

rf = data_catalog[f"tourism_rf_{q}_normal"].load()
plt.plot(rf["model"].pinball_loss_history_)
plt.plot(rf["model"].smooth_loss_history_)

# %%
min(rf["model"].pinball_loss_history_)

# %%

# %%
hist = true_y[:-60, 0]
from statsforecast.models import AutoARIMA

mdl = AutoARIMA(season_length=12)
mdl.fit(hist)

# %%

mdl.model_["residuals"].std()
# %%
