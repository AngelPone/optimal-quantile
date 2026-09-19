from pathlib import Path

from pytask import DataCatalog
import torch

SRC = Path(__file__).parent.resolve()
BLD = (Path(__file__).parent.parent.parent / "output").resolve()

TRAIN_WINDOWS = (2013 - 2006 + 1) * 12
VALID_WINDOWS = 12
TEST_WINDOWS = (2019 - 2015 + 1) * 12
WINDOW_S = (2005 - 1998 + 1) * 12
SAMPLE_SIZE = 500
OUTPUT_SAMPLE_SIZE = 10000
TOURISM_START = "1998-01"
ALPHAs = [0.05, 0.2, 0.8, 0.95]
BETA = 100

# Note that cpu can be extremely slow
DEVICE = torch.device("mps")  # use cuda is cuda is available
DTYPE = torch.float32


data_catalog = DataCatalog(name="tourism")


A = torch.concat(
    [
        torch.ones((1, 28)),
        torch.eye(4).tile(7),
        torch.kron(torch.eye(7), torch.ones(4)),
    ]
).to(dtype=DTYPE, device=DEVICE)
S = torch.concat([A, torch.eye(28, device=DEVICE, dtype=DTYPE)])
