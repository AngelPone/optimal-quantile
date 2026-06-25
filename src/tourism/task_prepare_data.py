from pytask import task, Product
import pandas as pd
import numpy as np
from tourism.config import (
    S,
    SRC,
    TRAIN_WINDOWS,
    TEST_WINDOWS,
    TOURISM_START,
    VALID_WINDOWS,
    WINDOW_S,
    data_catalog,
)
from typing import Annotated


@task
def task_prepare_data(
    node: Annotated[np.ndarray, Product] = data_catalog["tourism"],
):
    dt = pd.read_csv(SRC / "data" / "tourism.csv")
    dt = dt.loc[dt["Month"] >= TOURISM_START].reset_index(drop=True)
    values = dt.drop(columns=["Month"]).to_numpy(dtype=np.float64)

    expected_periods = WINDOW_S + TRAIN_WINDOWS + TEST_WINDOWS + VALID_WINDOWS
    if values.shape != (expected_periods, S.shape[0]):
        raise ValueError(
            "Tourism data does not match the configured empirical window: "
            f"{values.shape} != {(expected_periods, S.shape[0])}."
        )

    bottom = values[:, -S.shape[1] :]
    coherent = bottom @ S.numpy().T
    if not np.allclose(values, coherent, rtol=1e-10, atol=1e-4):
        raise ValueError("Tourism hierarchy is not coherent within rounding tolerance.")

    node.save(values)
