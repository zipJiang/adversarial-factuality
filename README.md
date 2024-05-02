# adversarial-factuality

### Usage

```shellscript

python scripts/run_task.py CONFIG_PATH [--cache-path CACHE_PATH]
```


### Configuration


See some examples in the `configs` directory. For example, `configs/dedupsoft_configs.yaml` runs a local model based FActScore alternative. To properly run the scorer you might need to download the wikipedia dump from [FActScore](https://github.com/shmsw25/FActScore)


### Sanity Check

(There's code update in between, so the results may not be exactly the same)

#### Pearson

| |mistral-7b-fs|mistral-7b-gs|mistral-7b-deberta-large|gpt-3.5-fs|gpt-3.5-fs-abs|
|---|---|---|---|---|---|
|mistral-7b-fs| |0.98|0.82|0.95|0.95|
|mistral-7b-gs|0.98| |0.83|0.93|0.93|
|mistral-7b-deberta-large|0.82|0.83| |0.73|0.73|
|gpt-3.5-fs|0.95|0.93|0.73| |1.00|
|gpt-3.5-fs-abs|0.95|0.93|0.73|1.00| |


#### Spearman

| |mistral-7b-fs|mistral-7b-gs|mistral-7b-deberta-large|gpt-3.5-fs|gpt-3.5-fs-abs|
|---|---|---|---|---|---|
|mistral-7b-fs| |0.97|0.94|0.79|0.79|
|mistral-7b-gs|0.97| |0.93|0.77|0.77|
|mistral-7b-deberta-large|0.94|0.93| |0.70|0.70|
|gpt-3.5-fs|0.79|0.77|0.70| |1.00|
|gpt-3.5-fs-abs|0.79|0.77|0.70|1.00| |

