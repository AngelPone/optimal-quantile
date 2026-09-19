# Experiments for Paper "Optimal Forecast Reconciliation for Quantiles"




## Prepare the environment

- Clone the repository  `git clone --depth=1 https://github.com/AngelPone/optimal-quantile`
- Install [uv](https://docs.astral.sh/uv/)
- Initialize the Python environment by  `uv sync`


## Run the tasks
```shell
# prepare M5 data
uv run src/M5/data_prepare.py --data-dir=/path/to/M5folder

# run tasks
uv run pytask
```


## Run the MCS and produce tables in the paper

```
Rscript src/simulation/mcs.R
Rscript src/tourism/mcs.R
Rscript src/M5_reduced/mcs.R
```
