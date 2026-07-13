from pathlib import Path
from pytask import DataCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLES_PATH = PROJECT_ROOT / "output" / "tables"
OUTPUT_PATH = PROJECT_ROOT / "output" / "M5_reduced"
DATA_OUTPUT_PATH = OUTPUT_PATH / "prepared.pkl"

TEST_WINDOWS = 1
FORECAST_HORIZON = 28
WINDOW_S = 52 * 7 * 3

ALPHAs = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
SAMPLE_SIZE = 200
OUTPUT_SAMPLE_SIZE = 10000
LR = 0.002
BETA = 100
data_catalog = DataCatalog(name="M5_reduced")
