NIGHT SHIFT - DEEP LEARNING ENDTERM
Askat Aubakirov and Alnur Shldenov
Group BDA2403

FILES TO SUBMIT
1. Endterm_BDA2403_Aubakirov_Shldenov.ipynb - executed notebook with saved outputs.
2. Endterm_BDA2403_Aubakirov_Shldenov.pdf - four report pages plus references.
3. Night_Shift_Demo.mp4 - 2 minutes 32 seconds of labelled recorded gameplay.
4. project_assets.zip - custom game source, portable trainers, weights and evidence.

GOOGLE COLAB
Open the notebook in Google Colab and choose Runtime > Run all.
The notebook downloads the assets from a fixed GitHub commit and checks SHA256.
No paid API, Drive mounting or training from scratch is required.
If the download is unavailable, upload project_assets.zip when prompted.
The default run shows the saved experiments and repeats 10 already tested cases.
It does not rerun the 300k/400k training experiments.

LOCAL JUPYTER
Keep project_assets.zip next to the notebook and run all cells.
The notebook extracts the assets automatically if needed.
CPU and CUDA runs were checked in fresh local kernels. The public download
was tested from an empty working folder. An actual hosted Colab session
was not accessible during verification; the setup is written for Colab.
Allow a few minutes for the replay check. Software/hardware changes can alter trajectories.

OPTIONAL FULL TRAINING
The notebook has separate flags for short teaching runs, full V1 B/C runs
from scratch and full V2 adaptation from the included V1 weights.
All flags are False by default. Full training can take hours.
Original V1 source uses the Gym registry ID NightShift-v0; V1 is the game version label.
Do not import V1 and V2 game modules into the same notebook kernel.
The full training commands run separate Python processes for this reason.

MAIN RECORDED RESULTS
V1 changed starts: B250k 33/40, C300k 36/40, on their V1 held-out protocol.
V2 matched changed starts: B original 32/40 -> latest 400k 13/40;
C original 29/40 -> selected 400k 38/40. Final choice: C 400k.
V1 and V2 test sets differ. Fixed repeats are one deterministic scenario.
B's best V2 diagnostic policy remains its original weights, not its latest 400k.
The missing full B V2 history is stated explicitly. Its received weights,
diagnostics and locally measured test results are preserved.

The report and notebook contain the methodological references and limitations.
The notebook and source files are not submitted to Moodle automatically.
