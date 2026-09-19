from M5_reduced.config import data_catalog, LOGGING_PATH

data_catalog["test_data"].load()["m5_weights"].to_csv(LOGGING_PATH / "m5_weights.csv")
