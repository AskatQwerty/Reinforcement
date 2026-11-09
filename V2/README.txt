NIGHT SHIFT V2 - TWO INDEPENDENT PYTORCH EXPERIMENTS

1. What changed
V1 is preserved separately. V2 changes one capture rule: while a police car
remains nearby, briefly accelerating no longer decreases the capture meter.
Capture still grows within 72 px when player speed is below 65 px/s.
It decreases only when ALL police cars are at least 120 px away; otherwise
it is held. The 3.2-second capture time, police speed, driving physics,
45-second exit unlock, 60 observations and 15 actions remain the same.

The motivation is the V1 agent's repeated close-pursuit acceleration that
reduced the capture meter without a real escape. This is a behavior observed
in traces, not a claim that the network consciously intended to cheat.

2. Which notebook to use
Main laptop: Train_C.ipynb, starting from C at 300,000 V1 steps.
Second laptop: Train_B.ipynb, starting from B at 250,000 V1 steps.
Copy this ENTIRE V2 folder to the other laptop before running anything.
Each laptop keeps its own runs folder. Do not synchronize live checkpoint files.
The initial weights in init are small; V1 replay and optimizer are not reused.

Open the intended notebook with a NEW kernel in the deep-learning environment.
This prevents Python from reusing a previously imported V1 environment.
Run All prepares the notebook and displays saved results; it does not train.
Click Start/resume V2 training when ready. The common target is 400,000
ADDITIONAL V2 environment steps. Resuming reaches that target, not another 400k.
At 50 steps/s, training alone takes about 133 minutes. Actual speed can differ.

3. Dependencies
The main laptop already has CUDA-enabled PyTorch. Keep that installation.
In the chosen Python environment, install the other requirements if needed:
python -m pip install -r requirements-training.txt
If the second laptop has no PyTorch, use the installation command from
https://pytorch.org/get-started/locally/ for its operating system and CUDA setup.
CPU is supported, but check throughput before promising a time budget.
The notebook prints the actual Python executable, PyTorch version and device.

4. Training and interruptions
Both runs use the same V2 rules and stable Double DQN settings. B uses 64, 64;
C uses 128, 128, 64 hidden units. Both retain potential-based goal shaping.
Training starts are 50% the exact original start and 50% small random changes
(x/y within 30 px, angle within 10 degrees), using a saved local random generator.
This start variation is an explicit training change shared by both models.

Checkpoints are saved every 10,000 steps and immutable snapshots every 50,000.
The replay, target network, optimizer, environment, RNG and timing are saved.
Use Kernel -> Interrupt Kernel to stop, then wait for the callback to finish.
A partial update is discarded; up to 10,000 steps after the last coherent
checkpoint can be lost. Run All and click Start/resume again to continue.
Each attempt keeps a separate timing/status JSON. Old attempts are retained.
If settings or source files change, the runner refuses to silently resume.

5. Evaluation
The baseline uses frozen V1 weights in V2:10 fixed-start repeats and 10 changed
starts, with original V2 reward and epsilon 0. It does not update any weights.
The package may already contain baseline outputs. Repeating the baseline
loads/resumes the matching evaluation attempt instead of erasing it.

At step 0 and every 50k training steps, a small diagnostic checks the fixed start
and four predefined variations. After 400k, the final evaluator chooses a saved snapshot
using ONLY these diagnostics: fixed-start success, then diagnostic success
rate, mean original reward, then later step. The final test never selects weights.
Step 0 is eligible: if adaptation makes the policy worse, the original policy
can be retained honestly rather than forcing a later, weaker checkpoint.
The report keeps trained_step and evaluation_step separate because a selected
snapshot can be earlier than the full training budget.

Final evaluation runs 10 fixed repeats and 40 NEW small start changes, then
records one full fixed-start video. Fixed repeats are deterministic; ten
identical wins do not demonstrate success on ten different situations.
Tests use original rewards, while training also uses the shaping reward.
Incomplete tests are saved and resume at the next unfinished episode.
Videos are separate demonstrations and do not enter the test averages.

After the trained policy's final test, the same button also evaluates the
ORIGINAL V1 weights on exactly the same 10+40 final V2 cases. This separate
reference phase makes before/after comparisons fair. It runs only after the
400k budget and diagnostic are complete, so final test starts are not inspected
before training. The earlier initial preview used different changed starts.
The reference makes no weight updates and records no extra video. If interrupted,
click Final checks again: completed final tests are reused and reference resumes.

For the clean final comparison, copy the completed runs/v2_b folder from
the second laptop into this V2 folder on the main laptop (keep both sources).
Then run final evaluations for B and C on the MAIN laptop. Compare only equal
additional training budgets and matching game/test versions. Evaluation files
from a different device or version are kept as separate attempts.

6. Saved files
init/manifest.json: origin, pretraining budget and hash of each initial policy.
runs/v2_b and runs/v2_c: checkpoints, diagnostic checks, histories and attempts.
evaluations/initial_B|C: frozen-policy baseline attempts.
evaluations/reference_B|C: original weights on the matched final V2 cases.
evaluations/final_B|C: final selected-policy tests, protocols and videos.
train_v2.py: explicit PyTorch Double DQN and persistence code.
eval_v2.py: frozen-policy evaluation and read-only plotting helpers.

V1 was trained for different budgets before these warm starts: B 250k, C 300k.
Therefore V2 compares adaptation of two saved policies, not an isolated proof
that one architecture is always better. The equal-budget V1 comparison is
preserved in the original experiment reports. Training only one seed per
configuration and testing small changes of one map remain limitations.
