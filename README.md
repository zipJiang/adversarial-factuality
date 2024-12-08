# Stepwise Core (And other Decompose-Then-Verify)

This refactorization of the `Core` codebase allows step-wise implementation of the decompose-then-verify pipeline. Building upon `tasker`, we can now run the pipeline in a more modular way, allowing for easier debugging and development, as well as manual examination of the intermediate results.

## Usage

```shellscript
# Make sure that the project directory is in your PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/run_task.py CONFIG_PATH
```


## Configuration

Instead of setting configuration files for a full pipeline in a single file, we now have a more modular configuration system. Each module in the pipeline has its own configuration file, and the main configuration file specifies the path to each module's configuration file. Each configuration file is pretty standard to `tasker`'s configuration file, which follows the following format:

```yaml
type: "module-name"
# configurations for parameters goes here
output_dir: "A directory to store the output"
dependencies:
    "dependency1": "path/to/dependency1/config"
    "dependency2": "path/to/dependency2/config"
    ...
```

It is very straightforward to set up the configuration files for each module. This implicitly defines the order of the pipeline, and the `tasker` will automatically run the pipeline in the correct order based on the dependencies, i.e. the module will only run after all its dependencies have finished running. And if any dependency is not up-to-date, the module will be rerun. That makes it easy as one only have to issue the command to run the final module, and the `tasker` will automatically run all the dependencies.


## Steps Overview

The pipeline is divided into the following steps:

1. **Abstention Detection**: This module detects abstention in the dataset. It is a standalone module that can be run independently of the rest of the pipeline. It outputs a file that contains the indices of the abstained samples. This can be run with configuration to `tasks/abstention_detection_task.py`.
2. **Decomposition**: This module decomposes the dataset into a set of sub-datasets. It is a standalone module that can be run independently of the rest of the pipeline. It outputs a set of files, each containing a sub-dataset. This can be run with configuration to `tasks/decomposition_task.py`.
3. **Post Processing**: This module performs post-processing on the decomposed datasets. It is a standalone module that can be run independently of the rest of the pipeline. It outputs a set of files, each containing a post-processed sub-dataset. This can be run with configuration to `tasks/post_processing_task.py`. Notice that it is possible to have multiple `post_processing` steps, so intermediate outputs can be used as inputs for the next `post_processing` step.
4. **Verification**: This module verifies the decomposed datasets. It is a standalone module that can be run independently of the rest of the pipeline. It outputs a set of files, each containing a verified sub-dataset. This can be run with configuration to `tasks/verification_task.py`.

## Data Flow

To make the pipeline more general, we keep the data flow flexible. The input data structure for `Abstention Detection` and `Decomposition` is a list of dictionaries, where each dictionary represents a sample:

```json
{
    "id_": "sample_id",
    "generation": "generation_id",
    "meta": {...}
    ...
}
```

Notice that any other fields not specified will be folded into `meta`. Each module may require specify fields in the input data structure (something one can always tuck into `meta`), and the output data structure from **Decomposition Step** is always going to be a list of dictionaries, where each dictionary represents a sample:

```json
{
    "id_": "sample_id",
    "generation": "generation_id",
    "meta": {...},
    "claims": [
        {
            "claim": "claim_text",
            "meta": {...}
            ...
        },
        ...
    ]
    ...
}
```

Where the `claims` field is serialized `AtomicClaim` that also admits `meta` field. The output data structure from **Verification Task** is going to be a list of dictionaries with two additional fields:

```json
{
    "id_": "sample_id",
    "generation": "generation_id",
    "meta": {...},
    "aggregated_score": "aggregated_score",
    "claims": [
        {
            "claim": "claim_text",
            "factual_score": "factual_score",
            "meta": {...}
            ...
        },
        ...
    ]
    ...
}
```

Thus, by plugging in your own data before any **Post Processing** step, you can easily adapt the pipeline to your own data structure.

## Visualization

Once you get the output from the **full pipeline** (Verification Task), you can visualize the result using the `scripts/streamlit_viewer.py` script.

```shellscript
streamlit run scripts/streamlit_viewer.py -- --data-dir ${OUTPUT_DIR}
```

Where `${OUTPUT_DIR}` is the directory where the output of the **full pipeline** is stored. The script will automatically load the data and display the result in a streamlit app.

![Streamlit Viewer](./media/visualization.png)

## Example Run

The configuration files in the `configs` directory should enable running a pre-configured pipeline over the `generations.jsonl` file in the `examples` directory.