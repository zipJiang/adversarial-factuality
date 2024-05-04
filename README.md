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


### Dataset for Sampling Generation

Since there were only ~200 datapoints in the FActScore dataset, we run the filtering over all the data from [nkandpa2/pretraining_entities](https://huggingface.co/datasets/nkandpa2/pretraining_entities) as well as the [popQA](https://github.com/AlexTMallen/adaptive-retrieval), aligning popularity metrics from both sources using Wikidata entity ids, resulting in the following dump format:

```json
{
    "entity_id": "Q189729", // entity id of wikidata
    "wikidata_freq": 1294, // number of entity mention from wikidata
    "popqa_freq": 45173, // popularity indicator from popQA
    "popqa_entity_name": "Philip Glass", // entity name from popQA
    "wikipedia_page": "https://en.wikipedia.org/wiki/Philip_Glass", // wikipedia page
    "wikipedia_title": "Philip_Glass", // wikipedia title
    "is_in_dump": true, // whether this entity can be queried in the FActScore-provided wikipedia dump
    "adjusted_freq": 45173, // adjusted frequency, max(popqa_freq, wikidata_freq)
    "adjusted_freq_source": "popqa", // either "popqa" or "wikidata", the source of adjusted frequency
    "already_selected": false // whether this entity has been selected for the FActScore dataset
}
```