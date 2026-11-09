# Night Shift: reinforcement learning

A student PyTorch project: learn to escape police in a custom top-down driving game.
The environment uses 60 numerical observations and 15 discrete actions. Training
uses a self-implemented Double DQN with replay, a target network and goal shaping.

## Current experiment: V2

V1 allowed a brief acceleration near police to decrease the capture meter. V2
keeps the meter while police remain nearby; it decreases only when every police
car is at least 120 pixels away. Driving physics, police speed and the 45-second
exit opening remain the same.

The full V1 archive and previous training history are retained on the main
laptop. This repository contains the small, portable V2 package, initial policy
weights and frozen-policy baseline results.

| Laptop | Notebook | Starting policy | Additional V2 training |
|---|---|---|---:|
| Main | [Train_C.ipynb](V2/Train_C.ipynb) | C after 300,000 V1 steps | 400,000 steps |
| Second | [Train_B.ipynb](V2/Train_B.ipynb) | B after 250,000 V1 steps | 400,000 steps |

## Start on the second laptop

Read [the complete instructions](V2/README.txt), or give a coding agent the
[Russian handoff prompt](SECOND_LAPTOP_PROMPT_RU.txt).

1. Clone this repository. Open `V2/Train_B.ipynb` with a new Python kernel.
2. Keep an existing working CUDA-enabled PyTorch installation. Install missing
   supporting packages with `python -m pip install -r V2/requirements-training.txt`.
   The exact versions used on the main laptop are recorded in
   [ENVIRONMENT_USED.json](V2/ENVIRONMENT_USED.json).
3. Run all notebook cells. They load weights, show saved results and create
   buttons; they do not start training automatically.
4. Click **Start/resume V2 training**. The target is 400,000 total additional V2
   steps, including any steps already saved in this run.

The same training can be launched from a terminal with the chosen Python:

```sh
python -B V2/tests_v2.py
python -B V2/train_v2.py --model B --steps 400000
```

Checkpoints are saved every 10,000 steps, with immutable snapshots and diagnostics
every 50,000. An interruption resumes from the last coherent checkpoint; up to
10,000 recent steps may need to be repeated. Training attempts and timings remain
separate. Do not change source files, hyperparameters or initial weights during a
run: checkpoint loading checks their hashes. `.gitattributes` preserves these
bytes when cloning on Windows.

## Return results

When B reaches 400,000 V2 steps, copy or archive its entire `V2/runs/v2_b` folder
for the main laptop. Keep the original copy. Large checkpoints, videos and ZIP
archives are ignored by Git. Final tests of both models should run on the main
laptop, using the same device, test starts and reward definition.

Final evaluation selects a snapshot using diagnostic results, then checks 10
fixed-start repeats and 40 new starts. It also evaluates the original policy on
the same final cases to provide a matched before/after comparison. Test outcomes
do not select the checkpoint. V1 pretraining budgets and V2 adaptation budgets
are reported separately.

## Files

- [train_v2.py](V2/train_v2.py): explicit Double DQN and resumable training.
- [eval_v2.py](V2/eval_v2.py): frozen-policy checks, videos and comparison.
- [gym_env.py](V2/gym_env.py): Gymnasium environment with `reset()` and `step()`.
- [init](V2/init): small initial policy weights and their provenance.
- [tests_v2.py](V2/tests_v2.py): capture, reset and environment regression checks.
- [QA_SUMMARY.json](V2/QA_SUMMARY.json): completed preparation checks.

Only one training seed per configuration is used. Fixed-start repetitions are
one deterministic scenario, and small changes of the start do not establish
performance on new maps. This V2 experiment compares adaptation from two saved
policies, not an isolated architecture comparison.
