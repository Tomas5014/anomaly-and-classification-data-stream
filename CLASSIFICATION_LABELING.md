# Active learning and delayed labeling for classification

The implementation is in `src/Classification/Labeling.py`. It operates only on
classification models and does not use the anomaly-detection pipeline.

## Experimental protocol

Every run treats the first benign block and the immediately following attack
region as initial supervised training. These instances are trained immediately,
without missing or delayed labels, and are excluded from evaluation metrics.
Benign interruptions of up to 1,000 instances do not split an attack region.
This matches the region convention already used by the project plots and keeps
the full first attack window in the timestamp-sorted scenarios.

After initial training:

- Experiment A uses full supervision and delays of 0%, 1%, 5%, and 10% of the
  total scenario size.
- Experiment B uses zero delay and Random Sampling probabilities of 1%, 3%, 5%,
  10%, 30%, and 100%.
- Experiment C uses the Cartesian product of the four delays and six sampling
  probabilities.

Random Sampling is Bernoulli sampling. Each evaluation instance independently
receives the configured probability of being selected. Percentages therefore
define an expected budget; the effective percentage is recorded for every run.
The same seed is reused across configurations, making lower budgets subsets of
higher budgets for a given run.

For a delay of `d`, a selected label from instance `t` becomes available before
prediction at instance `t + d`. With zero delay, training occurs after prediction
of the same instance. Labels whose delivery time is after the end of the stream
are flushed after evaluation and reported separately.

## Command-line execution

Preview one small run:

```bash
python run_classification_labeling.py \
  --categories Consistência \
  --sizes 25 \
  --models HT \
  --experiments A \
  --n-runs 1 \
  --dry-run
```

Run all three experiments for that scenario:

```bash
python run_classification_labeling.py \
  --categories Consistência \
  --sizes 25 \
  --models LB HAT ARF HT \
  --experiments A B C \
  --n-runs 5
```

Run `python run_classification_labeling.py --help` for dataset, feature-set,
window, output, plot, seed, and attack-gap options.

## Usage

Model entries must be factories that accept `run_seed`. The following example
uses the same factory convention as `Classification.ipynb`:

```python
from datetime import datetime

from src.Classification.Labeling import ClassificationLabelingExperimentRunner
from src.Classification.Models import get_classification_models

schema = stream.get_schema()

def make_ht(run_seed=None):
    return get_classification_models(
        schema,
        selected_models=["HT"],
        ht_params=default_params["HT"],
        run_seed=run_seed,
    )["HoeffdingTree"]

runner = ClassificationLabelingExperimentRunner(
    target_names=targets,
    n_runs=5,
    random_seed=42,
    attack_gap_tolerance=1000,
)

suite = runner.run_suite(
    stream=stream,
    algorithms={"HoeffdingTree": make_ht},
    experiments=("A", "B", "C"),
    window_evaluation=100,
    experiment_name=nome_experimento,
    scenario_name="Default_FullFeatures",
    exec_id=datetime.now().strftime("%Y%m%d_%H%M"),
    save_csv=True,
    generate_plots=False,
)

print(suite["paths"])
```

`run_suite` saves cumulative and prequential CSV files under
`output/ClassificationLabeling`. Equivalent configurations shared by experiments
A, B, and C are executed once and copied into each experiment's result rows.
Set `generate_plots=True` to create F1, precision, recall, FP, and FN plots for
every configuration.

## Public API

- `build_experiment_configs`: creates the A, B, and C grids.
- `calculate_delay_instances`: converts a delay fraction using total scenario
  size and nearest-integer rounding.
- `ClassificationLabelingExperimentRunner.prequential_test`: executes one model,
  seed, delay, and sampling probability.
- `ClassificationLabelingExperimentRunner.run_configuration`: aggregates runs
  for one configuration.
- `ClassificationLabelingExperimentRunner.run_suite`: executes and exports the
  requested experiment suite.
