# Thesis Experiment 3_4: Prompt-family Adaptive Search

## Task
Experiment 3 and Experiment 4 share the same audit harness.
Exp3 uses it to evaluate prompt-intervention search efficiency.
Exp4 uses the same harness with controlled targets: clone and fine-tuned variant.

## Methods
- Random
- JointPairUCB

## Environment
This experiment was run on the lab HPC with a separate Python environment because it uses HuggingFace / Transformers / Torch.

Recommended environment:
- Python 3.12
- torch
- transformers
- datasets
- pandas
- numpy
- matplotlib
- tqdm

## Run commands
For fair compute comparison, the Random baseline uses a larger explicit query budget.

JointPairUCB performs additional support-set evaluations during each adaptive update step, so the effective query cost of Random is approximately scaled to Q = 50 + 6t,
where t denotes the adaptive optimization step. The constant term corresponds to the initial exploration phase before bandit optimization.t.
### SST-2
```bash
python experiments/thesis/thesis_exp3_sst2.py \
  --seed 0,1,2,3,4 \
  --max_samples 200 \
  --search_budget 1250 \
  --top_k_holdout 1 \
  --methods Random


python experiments/thesis/thesis_exp3_sst2.py \
  --seed 0,1,2,3,4 \
  --max_samples 200 \
  --search_budget 200 \
  --top_k_holdout 1 \
  --methods JointPairUCB
```
### MNLI
```bash
python experiments/thesis/thesis_exp3_mnli.py \
  --seed 0,1,2,3,4 \
  --max_samples 200 \
  --search_budget 1250 \
  --top_k_holdout 1 \
  --methods Random

python experiments/thesis/thesis_exp3_mnli.py \
  --seed 0,1,2,3,4 \
  --max_samples 200 \
  --search_budget 200 \
  --top_k_holdout 1 \
  --methods JointPairUCB
```
## Output
Results are saved to:
results/exp23/sst_2/
results/exp23/mnli/

## Notes
Experiment 3 uses a different environment from Experiment 2 due to package version conflicts.
```text
requirements_exp2.txt
requirements_exp3.txt