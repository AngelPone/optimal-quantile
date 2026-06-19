from simulation.config import data_catalog, ALPHAs
from simulation.task_base_forecast import task_base_forecast
from simulation.task_reconciliation import task_perform_reconciliation
from simulation.task_collect import task_collect
from simulation.task_eval import task_evaluate

input_base = data_catalog[f"simulation_base_S2"].load()
input_data = data_catalog[f"simulation_S2"].load()

# task_base_forecast(input_data)

# task_perform_reconciliation(input_data, input_base, None, 0.95)

input_rf = data_catalog["simulation_rf_S2_0.8"].load()
task_collect(input_base, input_rf)

# input_samples = [data_catalog[f"simulation_samples_S1_{i}"].load() for i in ALPHAs]

# task_evaluate(input_data, input_samples)
