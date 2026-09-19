from pathlib import Path
from pytask import DataCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGGING_PATH = PROJECT_ROOT / "output" / "logs"
TABLES_PATH = PROJECT_ROOT / "output" / "tables"
OUTPUT_PATH = PROJECT_ROOT / "output" / "M5_reduced"
DATA_OUTPUT_PATH = OUTPUT_PATH / "prepared.pkl"

TEST_WINDOWS = 365
FORECAST_HORIZON = 28
WINDOW_S = 52 * 7 * 2

ALPHAs = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
OUTPUT_SAMPLE_SIZE = 5000
BETAs = [100]
data_catalog = DataCatalog(name="M5_reduced")
M5_PATH = Path("/Path/to/M5/dataset")
VERSION = 20260802
LR = 1e-4
SAMPLE_SIZE = {
    alpha: (1000 if alpha in [0.005, 0.025, 0.975, 0.995] else 500) for alpha in ALPHAs
}
VAL_SAMPLE_SIZE = 3000
MAX_ITER = 300
OUTSAMPLE_H = list(range(28))
DISTS = ["skew", "normal"]
INIT = "shr"
import torch

# Note that cpu can be extremely slow
DEVICE = torch.device("mps")  # use cuda is cuda is available
DTYPE = torch.float32
