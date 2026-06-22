from pathlib import Path
from pytask import DataCatalog
import torch

SRC = Path(__file__).parent.resolve()
BLD = (Path(__file__).parent.parent.parent / "output").resolve()

WINDOW_S = 850
TRAIN_WINDOWS = 100
TEST_WINDOWS = 50
DF = 6
SAMPLE_SIZE = 100
OUTPUT_SAMPLE_SIZE = 10000
data_catalog = DataCatalog(name="simulation")

A = torch.tensor(
    [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float64
)
SCENARIOS = ["S1", "S2", "S3", "S4"]
ALPHAs = [0.05, 0.2, 0.8, 0.95]
SIMULATION_SEED = 20260619


def scenario_seed(scenario: str, offset: int = 0) -> int:
    return SIMULATION_SEED + offset + 1_000 * SCENARIOS.index(scenario)


def scenario_alpha_seed(scenario: str, alpha: float, offset: int = 0) -> int:
    return scenario_seed(scenario, offset) + 100 * ALPHAs.index(alpha)
