"""Simple Double DQN adaptation. Importing this file never starts training."""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import gym_env
from night_env import GoalShaping

CONFIG={"code_version":1,"env_id":"NightShift-v2","seed":42,"gamma":0.995,
        "learning_rate":0.0001,"batch_size":64,"memory_capacity":50000,
        "warmup_steps":2000,"train_every":4,"target_update":1000,
        "epsilon_start":0.4,"epsilon_end":0.05,"epsilon_decay_steps":200000,
        "potential_scale":8.0,"save_every":10000,"evaluation_every":50000}
VALIDATION_CASES=[(0,0,0),(12,0,0),(-12,0,0),(0,20,5),(0,-20,-5)]
SOURCE_FILES=["train_v2.py","gym_env.py","simulation.py","handling.py","pursuit.py","night_env.py"]
_TRAINING_ACTIVE=False


class DQN(nn.Module):
    def __init__(self,input_size=60,output_size=15,hidden_sizes=(64,64)):
        super().__init__()
        layers=[]
        previous=input_size
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous,size),nn.ReLU()])
            previous=size
        layers.append(nn.Linear(previous,output_size))
        self.layers=nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


def now_text():
    return datetime.now(timezone.utc).isoformat()


def root_path(root=None):
    return Path(root).resolve() if root else Path(__file__).resolve().parent


def model_folder(model_name,root=None):
    if model_name not in ("B","C"):
        raise ValueError("Choose model B or C")
    return root_path(root)/"runs"/f"v2_{model_name.lower()}"


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def save_torch(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(data,temporary)
    temporary.replace(path)


def source_hashes(root):
    # A cached V1 module in an old notebook kernel must never run a V2 experiment.
    for name in ("gym_env","simulation","handling","pursuit","night_env"):
        if Path(sys.modules[name].__file__).resolve()!=root/f"{name}.py":
            raise RuntimeError("Restart the kernel and open the notebook from the V2 folder")
    return {name:file_hash(root/name) for name in SOURCE_FILES}


def initial_config(model_name,root):
    manifest=json.loads((root/"init"/"manifest.json").read_text(encoding="utf-8"))
    item=manifest["models"][model_name]
    path=root/"init"/item.get("weights",f"{model_name}_weights.pt")
    if file_hash(path)!=item["weights_sha256"]:
        raise ValueError("Initial weights do not match init/manifest.json")
    config=dict(CONFIG,model_name=model_name,hidden_sizes=item["hidden_sizes"],
                pretraining_steps=item["pretraining_steps"])
    return config,path,file_hash(root/"init"/"manifest.json")


def make_networks(config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy=DQN(hidden_sizes=config["hidden_sizes"]).to(device)
    target=DQN(hidden_sizes=config["hidden_sizes"]).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
    return policy,target,optimizer


def reset_training_episode(state):
    rng=state["starts_rng"]
    offset=(rng.uniform(-30,30),rng.uniform(-30,30),rng.uniform(-10,10)) if rng.random()<0.5 else (0,0,0)
    state["observation"],state["info"]=state["env"].reset(
        seed=rng.randrange(2**31),options={"start_offset":offset})
    state["episode_start"]=offset
    state["episode_reward"]=state["base_episode_reward"]=0.0
    state["episode_steps"]=0


def new_state(model_name,root):
    config,weights,init_hash=initial_config(model_name,root)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    policy,target,optimizer=make_networks(config)
    policy.load_state_dict(torch.load(weights,map_location=next(policy.parameters()).device,weights_only=True))
    target.load_state_dict(policy.state_dict())
    target.eval()
    state={"model_name":model_name,"config":config,"source_hashes":source_hashes(root),"init_hash":init_hash,
           "policy":policy,"target":target,"optimizer":optimizer,
           "env":GoalShaping(gym.make(config["env_id"]),scale=config["potential_scale"],gamma=config["gamma"]),
           "memory":deque(maxlen=config["memory_capacity"]),"step":0,
           "rng":random.Random(config["seed"]),"starts_rng":random.Random(config["seed"]+1),
           "episodes":[],"losses":[],"evaluations":[]}
    reset_training_episode(state)
    return state


def checkpoint_data(state):
    data={key:value for key,value in state.items() if key not in ("policy","target","optimizer")}
    data.update(policy=state["policy"].state_dict(),target=state["target"].state_dict(),
                optimizer=state["optimizer"].state_dict(),python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    return data


def load_checkpoint(model_name,root=None,checkpoint_path=None):
    root=root_path(root)
    path=Path(checkpoint_path) if checkpoint_path else model_folder(model_name,root)/"checkpoints"/"latest.pt"
    data=torch.load(path,map_location="cpu",weights_only=False)
    config,weights,init_hash=initial_config(model_name,root)
    if data["config"]!=config or data["source_hashes"]!=source_hashes(root) or data["init_hash"]!=init_hash:
        raise ValueError("This checkpoint belongs to different code, settings or initial weights")
    policy,target,optimizer=make_networks(config)
    policy.load_state_dict(data.pop("policy"))
    target.load_state_dict(data.pop("target"))
    target.eval()
    optimizer.load_state_dict(data.pop("optimizer"))
    device=next(policy.parameters()).device
    for values in optimizer.state.values():
        for key,value in values.items():
            # Adam's non-capturable step counter stays on the CPU.
            if isinstance(value,torch.Tensor) and key!="step":
                values[key]=value.to(device)
    random.setstate(data.pop("python_rng"))
    np.random.set_state(data.pop("numpy_rng"))
    torch.set_rng_state(data.pop("torch_rng"))
    cuda_rng=data.pop("cuda_rng")
    if device.type=="cuda" and cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    data.update(policy=policy,target=target,optimizer=optimizer)
    return data


def load_policy(model_name,checkpoint_path=None,root=None):
    root=root_path(root)
    config,weights,init_hash=initial_config(model_name,root)
    source_hashes(root)
    path=model_folder(model_name,root)/"checkpoints"/"latest.pt" if checkpoint_path is None else checkpoint_path
    if path=="initial" or (checkpoint_path is None and not Path(path).exists()):
        parameters=torch.load(weights,map_location="cpu",weights_only=True)
        config=dict(config,adaptation_steps=0)
    else:
        data=torch.load(Path(path),map_location="cpu",weights_only=False)
        if data["config"]!=config or data["source_hashes"]!=source_hashes(root) or data["init_hash"]!=init_hash:
            raise ValueError("Checkpoint configuration does not match this V2 package")
        parameters=data["policy"]
        config=dict(config,adaptation_steps=data["step"])
        data["env"].close()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy=DQN(hidden_sizes=config["hidden_sizes"]).to(device)
    policy.load_state_dict(parameters)
    return policy.eval().requires_grad_(False),config


def epsilon_at(step,config):
    fraction=min(1,step/config["epsilon_decay_steps"])
    return config["epsilon_start"]+(config["epsilon_end"]-config["epsilon_start"])*fraction


def train_one_step(state):
    config=state["config"]
    policy,target=state["policy"],state["target"]
    device=next(policy.parameters()).device
    observation=state["observation"]
    if state["rng"].random()<epsilon_at(state["step"],config):
        action=state["rng"].randrange(15)
    else:
        with torch.no_grad():
            action=policy(torch.as_tensor(observation,device=device).unsqueeze(0)).argmax(1).item()
    next_observation,reward,terminated,truncated,info=state["env"].step(action)
    state["memory"].append((observation,action,reward,next_observation,terminated))
    state["observation"],state["info"]=next_observation,info
    state["step"]+=1
    state["episode_steps"]+=1
    state["episode_reward"]+=reward
    state["base_episode_reward"]+=info["base_reward"]
    step=state["step"]
    if step>=config["warmup_steps"] and step%config["train_every"]==0:
        batch=state["rng"].sample(list(state["memory"]),config["batch_size"])
        states,actions,rewards,next_states,terminals=zip(*batch)
        states=torch.as_tensor(np.array(states),device=device)
        actions=torch.tensor(actions,dtype=torch.long,device=device)
        rewards=torch.tensor(rewards,dtype=torch.float32,device=device)
        next_states=torch.as_tensor(np.array(next_states),device=device)
        terminals=torch.tensor(terminals,dtype=torch.float32,device=device)
        predicted=policy(states).gather(1,actions[:,None]).squeeze(1)
        with torch.no_grad():
            next_actions=policy(next_states).argmax(1,keepdim=True)
            next_q=target(next_states).gather(1,next_actions).squeeze(1)
            expected=rewards+config["gamma"]*(1-terminals)*next_q
        loss=nn.functional.smooth_l1_loss(predicted,expected)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; resume from the last safe checkpoint")
        state["optimizer"].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(),10)
        state["optimizer"].step()
        state["losses"].append({"step":step,"loss":loss.item()})
    if step%config["target_update"]==0:
        target.load_state_dict(policy.state_dict())
    if terminated or truncated:
        state["episodes"].append({"step":step,"reward":state["episode_reward"],
            "base_reward":state["base_episode_reward"],"steps":state["episode_steps"],
            "seconds":info["elapsed"],"success":bool(info["success"]),"health":info["health"],
            "start_offset":state["episode_start"]})
        reset_training_episode(state)


def evaluate_state(state,cases=None,root=None):
    # Original rewards make all scores comparable. No training RNG is consumed.
    cases=VALIDATION_CASES if cases is None else cases
    model=state["policy"]
    device=next(model.parameters()).device
    env=gym.make(CONFIG["env_id"])
    rows=[]
    try:
        with torch.inference_mode():
            for i,case in enumerate(cases):
                observation,info=env.reset(seed=70000+i,options={"start_offset":case})
                total=0.0
                while True:
                    action=model(torch.as_tensor(observation,device=device).unsqueeze(0)).argmax(1).item()
                    observation,reward,terminated,truncated,info=env.step(action)
                    total+=reward
                    if terminated or truncated:
                        break
                rows.append({"reward":total,"success":bool(info["success"]),"seconds":info["elapsed"],
                             "health":info["health"],"case":case,"seed":70000+i})
    finally:
        env.close()
    return {"step":state["step"],"episodes":rows,"mean_reward":float(np.mean([x["reward"] for x in rows])),
            "std_reward":float(np.std([x["reward"] for x in rows])),
            "success_rate":float(np.mean([x["success"] for x in rows])),"fixed_success":rows[0]["success"]}


def show_status(model_name,root=None):
    folder=model_folder(model_name,root)
    path=folder/"status.json"
    result=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status":"not_started","saved_step":0}
    attempts=[json.loads(x.read_text(encoding="utf-8")) for x in sorted((folder/"attempts").glob("*.json"))]
    for category in ("training_seconds","evaluation_seconds","checkpoint_seconds"):
        result[category]=sum(x.get(category,0) for x in attempts)
    result["time_is_lower_bound"]=any(x.get("status")=="running" for x in attempts)
    print(f'{model_name}: {result["status"]}; saved V2 steps {result["saved_step"]:,}; '
          f'training {result["training_seconds"]/60:.1f} min')
    return result


def run_training(model_name,target_steps=400000,root=None):
    global _TRAINING_ACTIVE
    if _TRAINING_ACTIVE:
        raise RuntimeError("Training is already running in this kernel")
    _TRAINING_ACTIVE=True
    try:
        return _run_training(model_name,target_steps,root)
    finally:
        _TRAINING_ACTIVE=False


def _run_training(model_name,target_steps,root):
    if isinstance(target_steps,bool) or not isinstance(target_steps,int) or target_steps<1:
        raise ValueError("target_steps must be a positive absolute V2 step count")
    root=root_path(root)
    folder=model_folder(model_name,root)
    latest=folder/"checkpoints"/"latest.pt"
    torch.set_num_threads(1)
    state=load_checkpoint(model_name,root) if latest.exists() else new_state(model_name,root)
    if state["step"]>target_steps:
        state["env"].close()
        raise ValueError("Saved steps exceed this target; use a checkpoint for evaluation")
    folder.mkdir(parents=True,exist_ok=True)
    attempts=folder/"attempts"
    attempts.mkdir(exist_ok=True)
    numbers=[int(path.stem.split("_")[-1]) for path in attempts.glob("attempt_*.json")]
    attempt=attempts/f"attempt_{max(numbers,default=0)+1:03d}.json"
    record={"started_at":now_text(),"status":"running","start_step":state["step"],"target_steps":target_steps,
            "training_seconds":0.0,"evaluation_seconds":0.0,"checkpoint_seconds":0.0,"last_saved_step":state["step"]}
    started=time.perf_counter()
    completed=state["step"]

    def progress():
        write_json(attempt,record)
        write_json(folder/"status.json",{"model":model_name,"status":record["status"],
                   "saved_step":record["last_saved_step"],"target_steps":target_steps,"updated_at":now_text()})

    def checkpoint():
        before=time.perf_counter()
        try:
            save_torch(latest,checkpoint_data(state))
            record["last_saved_step"]=state["step"]
            snapshot=latest.parent/f'step_{state["step"]:07d}.pt'
            if state["step"]%CONFIG["evaluation_every"]==0 and not snapshot.exists():
                temporary=snapshot.with_suffix(".pt.tmp")
                shutil.copyfile(latest,temporary)
                temporary.replace(snapshot)
            save_torch(latest.parent/"latest_weights.pt",state["policy"].state_dict())
            write_json(folder/"history.json",{key:state[key] for key in ("episodes","losses","evaluations")})
        finally:
            record["checkpoint_seconds"]+=time.perf_counter()-before
        progress()

    def diagnostic():
        step=state["step"]
        if step%CONFIG["evaluation_every"]==0 and not any(x["step"]==step for x in state["evaluations"]):
            before=time.perf_counter()
            try:
                result=evaluate_state(state)
            finally:
                record["evaluation_seconds"]+=time.perf_counter()-before
            state["evaluations"].append(result)
            write_json(folder/"checks"/f"eval_{step:07d}.json",result)
            checkpoint()
            print(f'Validation: wins {result["success_rate"]:.0%}, fixed start {result["fixed_success"]}',flush=True)

    try:
        if not latest.exists():
            write_json(folder/"experiment.json",{"created_at":now_text(),"config":state["config"],
                       "source_hashes":state["source_hashes"],"init_hash":state["init_hash"],
                       "torch_version":torch.__version__,"gymnasium_version":gym.__version__,
                       "numpy_version":np.__version__,"device":str(next(state["policy"].parameters()).device),
                       "note":"V2 adaptation steps are additional to the saved V1 pretraining budget"})
            checkpoint()
        progress()
        diagnostic()
        print(f'{model_name}: resume at {state["step"]:,}; target {target_steps:,} additional V2 steps',flush=True)
        while state["step"]<target_steps:
            before=time.perf_counter()
            try:
                train_one_step(state)
            finally:
                record["training_seconds"]+=time.perf_counter()-before
            completed=state["step"]
            if completed%CONFIG["save_every"]==0:
                checkpoint()
                recent=state["episodes"][-20:]
                wins=np.mean([x["success"] for x in recent]) if recent else 0
                print(f'{completed:,}/{target_steps:,} | epsilon {epsilon_at(completed,CONFIG):.3f} | last20 wins {wins:.0%}',flush=True)
            diagnostic()
        checkpoint()
        record["status"]="complete"
    except KeyboardInterrupt:
        record["status"]="interrupted"
        print("Stopped. Resume uses the last complete checkpoint; up to10,000 recent steps may be repeated.",flush=True)
    except Exception as error:
        record.update(status="error",error=repr(error))
        raise
    finally:
        # Never publish the live state after an interrupted optimizer update.
        record.update(finished_at=now_text(),wall_seconds=time.perf_counter()-started,
                      last_completed_step=completed,unsaved_steps=max(0,completed-record["last_saved_step"]))
        progress()
        state["env"].close()
    return show_status(model_name,root)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["B","C"],required=True)
    parser.add_argument("--steps",type=int,default=400000)
    args=parser.parse_args()
    run_training(args.model,args.steps)
