from pathlib import Path

from pytask import DataCatalog
import torch

SRC = Path(__file__).parent.resolve()
BLD = (Path(__file__).parent.parent.parent / "output").resolve()

TRAIN_WINDOWS = (2013 - 2008 + 1) * 12
VALID_WINDOWS = 12
TEST_WINDOWS = (2019 - 2015 + 1) * 12
WINDOW_S = (2007 - 1998 + 1) * 12
DF = 6
SAMPLE_SIZE = 300
OUTPUT_SAMPLE_SIZE = 10000
TOURISM_START = "1998-01"
ALPHAs = [0.05, 0.2, 0.8, 0.95]
TOURISM_SEED = 20260623
BETA = 100
LR = 0.001

data_catalog = DataCatalog(name="tourism")


A = torch.concat(
    [
        torch.ones((1, 28)),
        torch.eye(4).tile(7),
        torch.kron(torch.eye(7), torch.ones(4)),
    ]
).to(dtype=torch.float64)
S = torch.concat([A, torch.eye(28)]).to(dtype=torch.float64)


def tourism_seed(offset: int = 0) -> int:
    return TOURISM_SEED + offset


def tourism_alpha_seed(alpha: float, offset: int = 0) -> int:
    return tourism_seed(offset) + 100 * ALPHAs.index(alpha)
