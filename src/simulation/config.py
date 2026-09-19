from pathlib import Path
from pytask import DataCatalog
import torch

SRC = Path(__file__).parent.resolve()
BLD = (Path(__file__).parent.parent.parent / "output").resolve()
LOG_PATH = BLD / "logs"

WINDOW_S = 200
TRAIN_WINDOWS = 400
TEST_WINDOWS = 400
DF = 8
SAMPLE_SIZE = 300
SKEWNESS_POS = 1.5
SKEWNESS_NEG = 0.66
OUTPUT_SAMPLE_SIZE = 10000
BETA = 100
data_catalog = DataCatalog(name="simulation")


SCENARIOS = ["S1", "S2", "S3", "S4"]
ALPHAs = [0.05, 0.2, 0.8, 0.95]
SIMULATION_SEED = 42

DTYPE = torch.float32
DEVICE = "mps"


def scenario_seed(scenario: str, offset: int = 0) -> int:
    return SIMULATION_SEED + offset + 1_000 * SCENARIOS.index(scenario)


def scenario_alpha_seed(scenario: str, alpha: float, offset: int = 0) -> int:
    return scenario_seed(scenario, offset) + 100 * ALPHAs.index(alpha)


A = torch.tensor(
    [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
    dtype=DTYPE,
    device=DEVICE,
)
