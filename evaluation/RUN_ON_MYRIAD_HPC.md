# Running the PolyMath eval on UCL Myriad (SGE)

Cluster docs: https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/

- Scheduler: **SGE** (`qsub`/`qstat`/`qdel`/`qacct`), not Slurm. Submitting/monitoring uses
  completely different commands from DIAS.
- Storage: `~/` (home, `myriadfs`) is **not backed up**, "at risk," 1TB quota — same story as
  DIAS. `~/ACFS` is backed up but **read-only from compute nodes**, so it's only useful for
  archiving finished results, not for the venv/container/model cache (which need write access
  during jobs).
- Myriad is a **heavily-shared general-purpose university cluster** (routinely 1500-2000+ jobs
  running/queued at once), unlike DIAS's largely-idle dedicated node. Expect real queue wait
  times, and don't assume a job's absence from `qstat -j` means it's finished — see the SGE
  quirks section below.
- GPU node types (`-ac allow=<code>`):
  - `EF` — V100s (compute capability 70). **Don't use these** — our model runs in `bfloat16`,
    which V100 doesn't support natively.
  - `L` — A100 40GB (compute capability 80).
  - `UV` — A100 80GB (compute capability 80). Preferred: matches DIAS's cards, and empirically
    had a free GPU when `L` was fully booked.

## 1. One-time setup (on the login node)

```bash
ssh myriad   # or ssh <ucl-id>@myriad.rc.ucl.ac.uk

git clone https://github.com/Rauljo/MuLE.git
cd MuLE
git checkout polymath_evaluation
```

### Git LFS

Same issue as DIAS: no `git-lfs` on Myriad, so `data-polymath/` comes through as pointer stubs
until fetched:

```bash
cd ~
curl -sL -o git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar xzf git-lfs.tar.gz
mkdir -p ~/bin
cp git-lfs-3.5.1/git-lfs ~/bin/
export PATH="$HOME/bin:$PATH"   # add to ~/.bashrc too
git lfs version

cd ~/MuLE
git lfs install --local
git lfs pull
head -c 4 evaluation/eval_tools/PolyMath/data-polymath/en/low.parquet   # should print PAR1
```

### Python environment (Apptainer)

Myriad actually has a real `python/3.11.x` module (unlike DIAS, which only had 3.9) — but per
the goal of a portable setup, this still uses Apptainer, exactly as on DIAS. `apptainer` is
already on `PATH` (no module load needed).

```bash
mkdir -p ~/containers
apptainer pull ~/containers/python311.sif docker://python:3.11   # non-slim: needs gcc for Triton JIT

apptainer exec ~/containers/python311.sif python3 -m venv ~/mule_venv
apptainer exec ~/containers/python311.sif bash -c "
  source ~/mule_venv/bin/activate
  pip install --upgrade pip
  pip install 'vllm==0.16.0'
  pip install -r ~/MuLE/evaluation/requirements-py311.txt
  pip install fasttext-wheel peft pandas seaborn matplotlib
"

mkdir -p ~/MuLE/evaluation/eval_tools/langid
curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
  -o ~/MuLE/evaluation/eval_tools/langid/lid.176.ftz
```

### Pre-download models

```bash
apptainer exec ~/containers/python311.sif bash -c "
  source ~/mule_venv/bin/activate
  python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    'XueZhang-bjtu/1.5B-cold-start-SFT',
    'XueZhang-bjtu/7B-cold-start-SFT',
    'XueZhang-bjtu/M-Thinker-1.5B-Iter1',
    'XueZhang-bjtu/M-Thinker-7B-Iter1',
    'kavmal7/lr_5e6_beta_0.2',
]:
    print(repo, '->', snapshot_download(repo))
PY
"
```

`kavmal7/lr_5e6_beta_0.2` is a **LoRA adapter**, not a full model (only `adapter_config.json` +
`adapter_model.safetensors`) — merge it onto the SFT base before vLLM can load it:

```bash
apptainer exec ~/containers/python311.sif bash -c "
  source ~/mule_venv/bin/activate
  python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = 'XueZhang-bjtu/1.5B-cold-start-SFT'
adapter = 'kavmal7/lr_5e6_beta_0.2'
out = '/home/<ucl-id>/merged_models/lr_5e6_beta_0.2'

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out)
tok.save_pretrained(out)
print('saved', out)
PY
"
```

## 2. SGE job script

Same `run_polymath_pipeline.sh` as DIAS — it's plain bash + Apptainer, scheduler-agnostic, no
changes needed. Only the wrapper job script differs (SGE `#$` directives instead of Slurm
`#SBATCH`):

```bash
#!/bin/bash -l
#$ -l h_rt=24:0:0
#$ -l mem=24G
#$ -l gpu=1
#$ -ac allow=UV
#$ -N mule_<model>
#$ -cwd
#$ -j y

SIF="$HOME/containers/python311.sif"
VENV="$HOME/mule_venv"
MODEL_PATH="<snapshot path or merged model path>"

ulimit -l unlimited
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv

apptainer exec --nv --cleanenv "$SIF" bash -c "
  export CC=gcc CXX=g++
  source \"$VENV/bin/activate\"
  MODEL_PATH=\"$MODEL_PATH\" MODEL_NAME=\"<name>\" GPU_MEM_UTIL=0.6 bash run_polymath_pipeline.sh
"
```

Submit and monitor:

```bash
cd ~/MuLE/evaluation
qsub run_myriad_<model>.qsub
qstat -u $USER
tail -f mule_<model>.o<jobid>       # combined stdout+stderr, thanks to -j y
```

To cancel: `qdel <jobid>`.

## 3. Analysis

Same as DIAS, no changes:

```bash
cd "PolyMath Answers Statistics"
apptainer exec ~/containers/python311.sif bash -c "
  source ~/mule_venv/bin/activate
  python3 Total_answers_stats.py --model_name <name>
  python3 Per_question_stats.py --model_name <name>
"
```

## Myriad-specific gotchas (not an issue on DIAS)

**`--cleanenv` is required, plus explicit `CC=gcc CXX=g++`.** Myriad's login/compute environment
sets `CC=icc CXX=icpc` by default (Intel compiler toolchain, loaded at login). Apptainer inherits
the host's environment by default, so `icc` leaks into the container — but our `python:3.11`
image only has `gcc` (no Intel compilers), so Triton's JIT compilation fails with
`FileNotFoundError: ... 'icc'`. Fix: `apptainer exec --nv --cleanenv ...` (strips host env
entirely) plus `export CC=gcc CXX=g++` inside the exec'd shell for extra safety. This never came
up on DIAS because that cluster's default environment doesn't set `CC`/`CXX` at all.

**`qalter` can't change `-l` (hard resource) requests on an already-submitted job** —
`jsv_allowed_mod` policy rejects it (`rejected due to jsv_allowed_mod configuration which does
not allow: l_hard`). To change memory/GPU/etc. requests, `qdel` and resubmit with a new script;
you can't tune a queued job in place like you (sometimes) can on other SGE/Slurm setups.

**Don't request a parallel environment (`-pe smp N`) alongside `-ac allow=UV`.** SGE resolves the
generic `smp` PE to a node-type-specific variant (`smp-U` for U-nodes, `smp-V` for V-nodes,
`smp-L` for L-nodes, etc.), and combining a *combined* access list (`UV`, matching either node
type) with a *single* generic PE request caused inconsistent resolution — the same node would
alternately report needing `smp-U` or `smp-V` depending on which queue instance was checked,
and neither matched. Since a single-GPU vLLM job doesn't need multiple CPU cores, the simplest
fix is to just not request a PE at all (1 slot, no `-pe` line).

**Use `qalter -w p <jobid>` to see exactly why a queued job hasn't started.** This is SGE's
scheduler dry-run/explain mode — it walks every candidate queue instance and states the specific
blocking reason (`offers only hc:gpu=0.000000`, `offers only hc:memory=4.000G`, `PE "smp-V" is
not in PE list`, `temporarily not available`, etc.). Far more actionable than just watching
`qstat` and guessing. Filter to the node types you actually care about, e.g.:
```bash
qalter -w p <jobid> 2>&1 | grep -i "node-v00a\|node-u00a" | sort -u
```

**A completed/deleted job can produce a false "job finished" signal from a single `qstat -j`
check before it's actually done** — a transient SSH/scheduler hiccup can make one `qstat -j
<jobid>` call fail even though the job is still genuinely queued or running. Confirm with a
second check (and look at whether the job's `.o<jobid>` output file actually exists /has grown)
before trusting a single "job not found" result.

**`qacct -j <jobid>` can return completely unrelated historical jobs.** SGE recycles job ID
numbers over the cluster's lifetime, and `qacct -j <id>` matches by number across *all* history —
on a long-lived cluster this returns other users' jobs from years earlier that happened to reuse
the same ID, not an error. Cross-check the `qsub_time`/`category` fields against your actual
submission to confirm you're looking at the right entry.

**Queue congestion is real and load-dependent, not a bug to "fix."** Unlike DIAS, Myriad had
~2000 jobs queued at once during testing. Use `qhost -F gpu | grep -A1 "node-<type>"` to check
live GPU availability per node type before assuming something is broken — a `qw` job sitting for
20+ minutes with a correctly-formed request is often just normal fair-share queueing, not an
error.
