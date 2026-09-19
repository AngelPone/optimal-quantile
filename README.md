
## Prepare the environment

- Install uv
```shell
uv sync
```

# M5

```shell
uv run src/M5/data_prepare.py --data-dir=/path/to/M5folder
```



1. different `h` what is the performance of different method
1. across h `h=1, h=7, h=28`, `alpha=0.835 0.975`
2. deep dive into Top-2 level
2. average G at different steps

# have done

Version20260728
1. use tensorboard to track the losses
3. learning rate decrease patience and cooldown

Version20260729: no substantial improvements
- increase sample size (3000 for tail quantile, 500 for middle quantiles) for quantiles at tails

Version20260730: start from ols (no longer search in local neibourhood of Shrinkage)
- results are worse, learning can not converge to the lowest loss as in the initial Shrinkage case.

Version20260731: start from ols, test the scheduler 
- slightly better than last version

Version20260731: start from ols, new scheduler (Constant + OneCycle) 
- worse 

Version20260733: start from ols, updated scheduler (Constant + OneCycle, smaller maximum learning rate)
- while well-trained on train set, not perform very well on test set


Version20260734: start from shr
- similar results to Version 20260728


Version20260735: start from wls
- better results for upper tails

Version20260736: set different learning rate for G and d
- Very slightly better (the gradients of d is very small)


Version20260737: add out-sample for different `h` use different out-sample residuals

- for outsample, base is much better, QOpt follows, shr is worse (for bigger h, base is better)
- for insample, results are similar with 20260736
- For skewnormal and normal, the results are very similar.

Version20260738: use SkewStudenT instead of Skewnormal, train a model for [0, 7, 14, 21] for out-of-sample