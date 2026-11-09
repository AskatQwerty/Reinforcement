"""Frozen-policy checks. Importing this module does not train or evaluate."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import gym_env
from train_v2 import load_policy


def root_path(root):
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parent


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_cases(count,seed):
    # These starts are separate from the five training diagnostic starts.
    rng=random.Random(seed)
    return [(rng.uniform(-30,30),rng.uniform(-30,30),rng.uniform(-10,10)) for _ in range(count)]


def summarize(rows):
    result={}
    for group in ("fixed","changed"):
        items=[x for x in rows if x["group"]==group]
        wins=[x for x in items if x["success"]]
        result[group]={"episodes":len(items),"wins":len(wins),
                       "success_rate":len(wins)/len(items) if items else None,
                       "mean_reward":float(np.mean([x["reward"] for x in items])) if items else None,
                       "std_reward":float(np.std([x["reward"] for x in items])) if items else None,
                       "mean_seconds":float(np.mean([x["seconds"] for x in items])) if items else None,
                       "mean_win_seconds":float(np.mean([x["seconds"] for x in wins])) if wins else None}
    return result


def one_episode(model,case,seed,video=None):
    env=gym.make("NightShift-v2")
    observation,info=env.reset(seed=seed,options={"start_offset":case})
    device=next(model.parameters()).device
    reward_sum=0.0
    steps=0
    max_capture=0.0
    writer=pg=None
    owns_display=False
    try:
        if video:
            import os
            import pygame as pg
            import imageio.v2 as imageio
            from main import Renderer,WIDTH,HEIGHT
            if pg.display.get_surface() is not None:
                raise RuntimeError("Close the manual game before recording a video.")
            previous_driver=os.environ.get("SDL_VIDEODRIVER")
            display_ready=pg.display.get_init()
            font_ready=pg.font.get_init()
            owns_display=True
            pg.display.quit()
            os.environ["SDL_VIDEODRIVER"]="dummy"
            pg.display.init()
            pg.font.init()
            screen=pg.display.set_mode((WIDTH,HEIGHT))
            renderer=Renderer(screen,env.unwrapped.game)
            writer=imageio.get_writer(str(video),fps=10,codec="libx264",quality=6)
        with torch.inference_mode():
            while True:
                state=torch.as_tensor(observation,dtype=torch.float32,device=device).unsqueeze(0)
                action=int(model(state).argmax(dim=1).item())
                observation,reward,terminated,truncated,info=env.step(action)
                reward_sum+=reward
                steps+=1
                max_capture=max(max_capture,info["capture"])
                if writer:
                    from gym_env import CONTROL_DT
                    game=env.unwrapped.game
                    renderer.scene(game,CONTROL_DT,True)
                    renderer.hud(game)
                    frame=pg.transform.smoothscale(renderer.frame,(640,400))
                    writer.append_data(pg.surfarray.array3d(frame).transpose(1,0,2))
                if terminated or truncated:
                    return {"reward":float(reward_sum),"success":bool(info["success"]),
                            "seconds":float(info["elapsed"]),"health":float(info["health"]),
                            "status":info["status"],"reason":info["reason"],"steps":steps,
                            "max_capture":float(max_capture),"terminated":bool(terminated),
                            "truncated":bool(truncated),"case":list(case),"seed":seed}
    finally:
        env.close()
        if writer:
            writer.close()
        if owns_display:
            pg.display.quit()
            if not font_ready:
                pg.font.quit()
            if previous_driver is None:
                os.environ.pop("SDL_VIDEODRIVER",None)
            else:
                os.environ["SDL_VIDEODRIVER"]=previous_driver
            if display_ready:
                pg.display.init()


def select_checkpoint(model_name,root):
    # Selection uses diagnostics, never the final held-out test outcomes.
    folder=root/"runs"/f"v2_{model_name.lower()}"
    history=json.loads((folder/"history.json").read_text(encoding="utf-8"))
    candidates=[]
    for result in history["evaluations"]:
        checkpoint=folder/"checkpoints"/f"step_{result['step']:07d}.pt"
        if checkpoint.exists():
            candidates.append((result,checkpoint))
    if not candidates:
        raise RuntimeError("No saved diagnostic checkpoints. Complete at least one 50,000-step block.")
    candidates.sort(key=lambda item:(bool(item[0]["fixed_success"]),item[0]["success_rate"],
                                     item[0]["mean_reward"],item[0]["step"]),reverse=True)
    return candidates[0][1],candidates[0][0]


def _evaluate(model_name,phase,root=None,record=True):
    model_name=model_name.upper()
    if model_name not in ("B","C"):
        raise ValueError("Use model B or C")
    if phase not in ("initial","final","reference"):
        raise ValueError("Unknown evaluation phase")
    root=root_path(root)
    if gym_env.ENV_ID!="NightShift-v2":
        raise RuntimeError("Restart the kernel and open the V2 notebook; V1 was imported earlier.")
    init=json.loads((root/"init"/"manifest.json").read_text(encoding="utf-8"))["models"][model_name]
    selected=None
    if phase=="initial":
        checkpoint=root/"init"/init["weights"]
        policy,config=load_policy(model_name,checkpoint_path="initial",root=root)
        trained_step=evaluation_step=0
        changed=test_cases(10,60261)
    else:
        # Validate the latest model, source and initial weights before trusting its step.
        latest_policy,latest_config=load_policy(model_name,root=root)
        trained_step=int(latest_config["adaptation_steps"])
        del latest_policy
        if trained_step<400000:
            raise RuntimeError(f"Final comparison needs 400,000 additional steps; saved {trained_step:,}.")
        history=json.loads((root/"runs"/f"v2_{model_name.lower()}"/"history.json").read_text(encoding="utf-8"))
        if not any(x["step"]==trained_step for x in history["evaluations"]):
            raise RuntimeError("The last diagnostic is unfinished. Resume training at the same step target first.")
        if phase=="reference":
            # Only after training: test the original weights on the final test cases.
            checkpoint=root/"init"/init["weights"]
            policy,config=load_policy(model_name,checkpoint_path="initial",root=root)
            evaluation_step=0
        else:
            checkpoint,selected=select_checkpoint(model_name,root)
            evaluation_step=int(selected["step"])
            policy,config=load_policy(model_name,checkpoint_path=checkpoint,root=root)
            if config["adaptation_steps"]!=evaluation_step:
                raise ValueError("The selected checkpoint does not match its diagnostic step.")
        changed=test_cases(40,60419)
    policy.eval().requires_grad_(False)
    cases=[{"group":"fixed","case":[0,0,0],"seed":80000+i} for i in range(10)]
    cases.extend({"group":"changed","case":list(case),"seed":81000+i} for i,case in enumerate(changed))
    sources={name:file_hash(root/name) for name in ("gym_env.py","simulation.py","pursuit.py","handling.py")}
    protocol={"version":1,"phase":phase,"model_name":model_name,"pretraining_steps":init["pretraining_steps"],
              "trained_step":trained_step,"evaluation_step":evaluation_step,"checkpoint":str(checkpoint.relative_to(root)),
              "checkpoint_sha256":file_hash(checkpoint),"game_hashes":sources,"cases":cases,
              "selection_rule":"fixed diagnostic win, diagnostic success rate, mean original reward, later step",
              "selection_diagnostic":selected,"device":str(next(policy.parameters()).device),
              "torch_version":torch.__version__,"numpy_version":np.__version__,
              "reward":"Original V2 reward; no goal shaping in evaluation","epsilon":0.0}
    if phase=="reference":
        protocol["selection_rule"]="Original V1 weights tested on the final V2 cases; no selection"
    base=root/"evaluations"/f"{phase}_{model_name}"
    attempts=sorted(base.glob("attempt_*/protocol.json"))
    folder=None
    for path in reversed(attempts):
        if json.loads(path.read_text(encoding="utf-8"))==protocol:
            folder=path.parent
            break
    if folder is None:
        folder=base/f"attempt_{len(attempts)+1:03d}"
        folder.mkdir(parents=True,exist_ok=False)
        write_json(folder/"protocol.json",protocol)
    result_path=folder/"result.json"
    result=json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "rows":[],"evaluation_seconds":0.0,"video_seconds":0.0,"video":None}
    result.update({"model_name":model_name,"phase":phase,"trained_step":trained_step,
                   "evaluation_step":evaluation_step,"pretraining_steps":init["pretraining_steps"],
                   "requested_episodes":len(cases),"status":"running"})
    started=time.perf_counter()
    previous_seconds=result["evaluation_seconds"]

    def save():
        result["summary"]=summarize(result["rows"])
        result["complete"]=len(result["rows"])==len(cases)
        write_json(result_path,result)

    try:
        save()
        for index in range(len(result["rows"]),len(cases)):
            case=cases[index]
            row=one_episode(policy,case["case"],case["seed"])
            row["group"]=case["group"]
            result["rows"].append(row)
            result["evaluation_seconds"]=previous_seconds+time.perf_counter()-started
            save()
            print(f'{phase} {model_name}: {index+1}/{len(cases)} | {row["status"]} | {row["seconds"]:.1f}s',flush=True)
        result["status"]="complete"
    except KeyboardInterrupt:
        result["status"]="interrupted"
        print("Stopped. Completed episodes are saved; run this evaluation again to continue.")
    except Exception as error:
        result["status"]="error"
        result["error"]=repr(error)
        raise
    finally:
        result["evaluation_seconds"]=previous_seconds+time.perf_counter()-started
        save()
    if result["complete"] and record and not result.get("video"):
        started=time.perf_counter()
        previous_video_seconds=result["video_seconds"]
        video=folder/f"preview_{len(list(folder.glob('preview_*.mp4')))+1:03d}.mp4"
        try:
            outcome=one_episode(policy,(0,0,0),80000,video)
            result["video"]=str(video.relative_to(root))
            result["video_outcome"]=outcome
        except KeyboardInterrupt:
            result["video_error"]="Video interrupted; tests remain complete. Run evaluation again to retry video."
        except Exception as error:
            result["video_error"]=repr(error)
        finally:
            result["video_seconds"]=previous_video_seconds+time.perf_counter()-started
            save()
    print("Saved:",result_path)
    return result


def safe_evaluation(model_name,phase,root,record):
    # Evaluation must not change the next random training action.
    python_rng=random.getstate()
    numpy_rng=np.random.get_state()
    torch_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return _evaluate(model_name,phase,root,record)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def evaluate_initial(model_name,root=None,record=False):
    return safe_evaluation(model_name,"initial",root,record)


def evaluate_final(model_name,root=None,record=True):
    result=safe_evaluation(model_name,"final",root,record)
    if result["complete"]:
        # A matched reference separates learning from differences in test starts.
        safe_evaluation(model_name,"reference",root,False)
    return result


def show_evaluations(model_name,root=None):
    root=root_path(root)
    results=[]
    for phase in ("initial","reference","final"):
        paths=sorted((root/"evaluations"/f"{phase}_{model_name.upper()}").glob("attempt_*/result.json"))
        if not paths:
            print(phase,": not evaluated")
            continue
        result=json.loads(paths[-1].read_text(encoding="utf-8"))
        results.append(result)
        print(f"{phase}: {result['status']} | trained +{result['trained_step']:,}, evaluated +{result['evaluation_step']:,}")
        if phase=="reference":
            print("  Original V1 weights on the final V2 cases; evaluated only after training.")
        for group,item in result["summary"].items():
            if item["episodes"]:
                print(f"  {group}: {item['wins']}/{item['episodes']} wins | reward {item['mean_reward']:.2f} +/- {item['std_reward']:.2f}")
    return results


def plot_training(model_name,root=None):
    import matplotlib.pyplot as plt
    root=root_path(root)
    path=root/"runs"/f"v2_{model_name.lower()}"/"history.json"
    if not path.exists():
        print("No V2 training history yet.")
        return
    history=json.loads(path.read_text(encoding="utf-8"))
    rows=history["episodes"]
    if not rows:
        print("No completed training episodes yet.")
        return
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    # Raw rewards and a short moving average show actual training progress.
    steps=[x["step"] for x in rows]
    rewards=[x["base_reward"] for x in rows]
    smooth=[float(np.mean(rewards[max(0,i-19):i+1])) for i in range(len(rows))]
    axes[0].plot(steps,rewards,alpha=.25,label="Episode")
    axes[0].plot(steps,smooth,label="Last 20")
    axes[0].set(xlabel="Additional V2 steps",ylabel="Original reward")
    axes[0].legend()
    wins=[float(np.mean([r["success"] for r in rows[max(0,i-19):i+1]])) for i in range(len(rows))]
    axes[1].plot(steps,wins)
    axes[1].set(xlabel="Additional V2 steps",ylabel="Win fraction: last 20",ylim=(0,1))
    fig.tight_layout()
    return fig


def compare_final(root=None):
    root=root_path(root)
    selected=[]
    matched_keys=("trained_step","game_hashes","cases","device","torch_version","numpy_version")
    for model_name in ("B","C"):
        phases={}
        for phase in ("reference","final"):
            paths=sorted((root/"evaluations"/f"{phase}_{model_name}").glob("attempt_*/result.json"))
            if not paths:
                raise RuntimeError("Run final evaluation for B and C, including their matched references, first.")
            result=json.loads(paths[-1].read_text(encoding="utf-8"))
            protocol=json.loads((paths[-1].parent/"protocol.json").read_text(encoding="utf-8"))
            if not result["complete"]:
                raise RuntimeError(f"{phase} {model_name} is incomplete. Run its final evaluation button to continue.")
            if result["model_name"]!=model_name or protocol["model_name"]!=model_name:
                raise RuntimeError("An evaluation folder contains another model's results.")
            phases[phase]=(result,protocol)
        for key in matched_keys:
            if phases["reference"][1][key]!=phases["final"][1][key]:
                raise RuntimeError(f"Model {model_name}: unmatched reference/final {key}.")
        selected.append(phases)
    for key in matched_keys:
        if selected[0]["final"][1][key]!=selected[1]["final"][1][key]:
            raise RuntimeError(f"Unmatched {key}; evaluate both models on the same computer and budget.")
    for phases in selected:
        result=phases["final"][0]
        reference=phases["reference"][0]
        print(f"Model {result['model_name']} | V1 {result['pretraining_steps']:,} + V2 {result['trained_step']:,} steps")
        for label,rows in (("Before: original V1 weights",reference),("After: selected V2 checkpoint",result)):
            print(" ",label)
            for group,item in rows["summary"].items():
                print(f"    {group}: {item['wins']}/{item['episodes']} wins; reward {item['mean_reward']:.2f} +/- {item['std_reward']:.2f}")
    return [dict(x["final"][0],reference_summary=x["reference"][0]["summary"]) for x in selected]
