from pathlib import Path
from pytask import DataCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGGING_PATH = PROJECT_ROOT / "output" / "logs"
TABLES_PATH = PROJECT_ROOT / "output" / "tables"
OUTPUT_PATH = PROJECT_ROOT / "output" / "M5_reduced"
DATA_OUTPUT_PATH = OUTPUT_PATH / "prepared.pkl"

TEST_WINDOWS = 1
FORECAST_HORIZON = 28
WINDOW_S = 52 * 7 * 3

ALPHAs = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
OUTPUT_SAMPLE_SIZE = 10000
BETAs = [100]
data_catalog = DataCatalog(name="M5_reduced")
M5_PATH = Path(
    "/Users/bohan/Library/CloudStorage/Nextcloud-bohan@nc․bohan-zhang․com/Documents/datasets/m5-forecasting-accuracy",
)
VERSION = 20260738
LR = 1e-5
SAMPLE_SIZE = {
    alpha: (3000 if alpha in [0.005, 0.025, 0.975, 0.995] else 500) for alpha in ALPHAs
}
VAL_SAMPLE_SIZE = 5000
MAX_ITER = 500
OUTSAMPLE_H = [0, 7, 14, 21]
DISTS = ["skew", "normal"]

import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float64
