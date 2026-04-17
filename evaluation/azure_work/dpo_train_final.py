import os

# Set writable cache directories BEFORE any imports that use datasets/transformers
# This prevents "Read-only file system" errors when loading from Azure ML inputs
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)
os.makedirs("/tmp/hf_datasets_cache", exist_ok=True)

# W&B and HF home directories (writable locations for cloud/cluster environments)
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["WANDB_CONFIG_DIR"] = "/tmp/wandb_config"
os.makedirs("/tmp/hf_home", exist_ok=True)
os.makedirs("/tmp/wandb_config", exist_ok=True)

os.system("pip install datasets transformers unsloth trl bitsandbytes wandb")
import json
import argparse
import logging
import warnings
import gc
 
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
 
import torch
import wandb
import datasets
from datasets import Dataset, load_from_disk

# Explicitly set datasets cache to writable location
datasets.config.HF_DATASETS_CACHE = "/tmp/hf_datasets_cache"
datasets.config.IN_MEMORY_MAX_SIZE = 0  # Force disk caching
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
 
from transformers import TrainerCallback
import torch
import time

 
MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"
 
# ─────────────────────────────────────────────
# CONFIG — edit these values to match your setup
#
# Colab:   set paths to "/content/your_file.jsonl"
# VS Code: set paths to wherever your files are
# Cluster: set full absolute paths
# ─────────────────────────────────────────────
CONFIG = {
    "dataset_file":     "./dpo_dataset_demo.jsonl",
    "stats_file":       "./stats.json",
    "output_dir":       "./dpo_output",
 
    "wandb_project":    "mule-dpo",
    "wandb_run_name":   "dpo-run1",
 
    # Hyperparameters matched to M-Thinker SFT 1.5B (Appendix B.3)
    # SFT used: lr=1e-6, batch=256, epochs=1
    "beta":             0.1,
    "learning_rate":    1e-6,
    "num_epochs":       1,
    "batch_size":       1,
    "grad_accum_steps": 64,  # effective batch = 64, M-Thinker ST used 256, but this way you same VRAM and get more andb updates
 
    # max_length=4096: reduced from M-Thinker's 16384 due to RTX 6000 (23GB)
    # memory constraints. truncation_mode=keep_end preserves the \boxed{} answer.
    "max_length":       16384,
 
    # max_pairs: how many pairs to use from the full dataset.
    # Full dataset has ~7785 pairs but precompute takes 7 hours for all of them.
    # 1000 pairs = ~1 hour total (precompute + training). Scientifically valid subset.
    # Set to None to use all pairs (only do this if you have many hours available).
    "max_pairs":        None,
 
    # cpu_test=True:  1 pair, 1 epoch, max_length=256 for local testing
    # cpu_test=False: full training on GPU
    "cpu_test":         False,
}
 
 
def load_dpo_dataset(dataset_file, cpu_test=False, max_pairs=None):
    """
    Load preference pairs from dpo_dataset.jsonl.
 
    max_pairs: if set, takes the N shortest pairs from the dataset.
               Shortest = least truncation at max_length=4096.
               This makes precompute faster while keeping training signal clean.
    """
    pairs = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "prompt" not in record or "chosen" not in record or "rejected" not in record:
                    logging.warning(f"Line {line_num}: missing field, skipping.")
                    continue
                pairs.append(record)
            except json.JSONDecodeError as e:
                logging.warning(f"Line {line_num}: JSON error ({e}), skipping.")
                continue
 
    total_loaded = len(pairs)
 
    if cpu_test:
        # 1 shortest pair for CPU test
        pairs.sort(key=lambda p: len(p["prompt"]) + len(p["chosen"]) + len(p["rejected"]))
        pairs = pairs[:1]
        logging.info("CPU TEST MODE: using only the 1 shortest pair")
    elif max_pairs is not None and max_pairs < total_loaded:
        # Take the N shortest pairs — these benefit most from training
        # since they fit within max_length with minimal truncation
        pairs.sort(key=lambda p: len(p["prompt"]) + len(p["chosen"]) + len(p["rejected"]))
        pairs = pairs[:max_pairs]
        logging.info(f"Using {max_pairs} shortest pairs out of {total_loaded} total")
 
    return pairs
 
 
def load_stats(stats_file):
    if not os.path.exists(stats_file):
        logging.warning(f"Stats file not found: {stats_file}")
        return {}
    with open(stats_file, "r") as f:
        return json.load(f)
    
def get_latest_checkpoint(output_dir):
    if output_dir is None:
    	return None
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None

    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(output_dir, latest)
 

class GPUMemoryWatchdog(TrainerCallback):
    def __init__(self,  check_interval=1):
        """
        check_interval: check every N steps
        """
        self.check_interval = check_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_interval != 0:
            return control

        if not torch.cuda.is_available():
            return control

        free, total = torch.cuda.mem_get_info()
        used = total - free
        utilization = used / total

        print(f"\nGPU memory usage: {utilization*100:.1f}% used")

        return control
    
def main(args):        
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
 
    if args.cpu_test:
        logger.info("=" * 55)
        logger.info("  CPU TEST MODE")
        logger.info("  1 pair, 1 epoch, max_length=256")
        logger.info("=" * 55)
    else:
        logger.info("=" * 55)
        logger.info("  FULL TRAINING MODE")
        logger.info(f"  max_pairs={args.max_pairs}, max_length={args.max_length}")
        logger.info("=" * 55)
 
    # W&B initialisation
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    logger.info(f"W&B run started: {args.wandb_run_name}")

    # Load model and tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    PatchDPOTrainer()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = args.max_length,
        load_in_4bit = True,
        attn_implementation = "sdpa", # Use sdpa instead of xformers
        dtype = torch.bfloat16,      # Use BF16 instead of FP16/None
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", # Optimized for long context
        
    )
    model = FastLanguageModel.for_training(model)
    model.config.use_cache = False  # required for gradient checkpointing
    
    # Load pairs
    logger.info(f"Loading DPO pairs from: {args.dataset_file}")
    max_pairs = None if args.cpu_test else args.max_pairs
    #pairs = load_dpo_dataset(args.dataset_file, cpu_test=args.cpu_test, max_pairs=max_pairs)
    # Change how you load the dataset

        
    logger.info(f"Loading precomputed dataset from disk: {args.dataset_file}")
    
    # Copy dataset to writable location to avoid read-only file system errors
    import shutil
    writable_dataset_path = "/tmp/dataset_copy"
    if os.path.exists(writable_dataset_path):
        shutil.rmtree(writable_dataset_path)
    shutil.copytree(args.dataset_file, writable_dataset_path)
    
    train_dataset = load_from_disk(writable_dataset_path)
    if max_pairs and max_pairs < len(train_dataset):
        train_dataset = train_dataset.select(range(max_pairs))
    if len(train_dataset) == 0:
        raise ValueError("No pairs loaded from precomputed dataset.")
    # Normalise column names: precompute_ref.py writes reference_{key}_logps
    # but TRL's DPOTrainer expects ref_{key}_logps.
    for old, new in [("reference_chosen_logps",   "ref_chosen_logps"),
                     ("reference_rejected_logps", "ref_rejected_logps")]:
        if old in train_dataset.column_names:
            train_dataset = train_dataset.rename_column(old, new)
    ref_model = None
    precompute_ref_log_probs = False
 
    
    logger.info(f"Loaded {len(train_dataset)} preference pairs.")

    logger.info("--- Sanity check: first pair ---")
    first = train_dataset[0]
    logger.info(f"PROMPT   (first 150 chars): {first['prompt'][:150]}")
    logger.info(f"CHOSEN   (first 150 chars): {first['chosen'][:150]}")
    logger.info(f"REJECTED (first 150 chars): {first['rejected'][:150]}")
 
    # Load and log dataset stats to W&B summary
    stats = load_stats(args.stats_file)
    if stats:
        for k, v in stats.items():
            if not isinstance(v, list):
                wandb.run.summary[f"dataset/{k}"] = v
        if stats.get("total_questions", 0) > 0:
            wandb.run.summary["dataset/pair_yield_pct"] = (
                100 * stats.get("pairs_created", 0) / stats["total_questions"]
            )
 
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Train: {len(train_dataset)} pairs")
 
    wandb.run.summary["dataset/train_pairs"] = len(train_dataset)
    wandb.run.summary["dataset/eval_pairs"]  = 0
    logger.info("Dataset stats logged to W&B summary")
 
    # CPU test vs GPU settings
    num_epochs = 1    if args.cpu_test else args.num_epochs
    max_length  = 256 if args.cpu_test else args.max_length
    grad_accum  = 1   if args.cpu_test else args.grad_accum_steps
 
    logger.info(f"Config: epochs={num_epochs}, max_length={max_length}, grad_accum={grad_accum}")
 
    # Log GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb  = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        logger.info(f"GPU memory at start: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
 
    
 
    # Log GPU memory after model load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb  = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        logger.info(f"GPU memory after model load: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
 
    # DPO Config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=num_epochs,
 
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
 
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
 
        beta=args.beta,
        loss_type="sigmoid",
 
        max_length=max_length,
        truncation_mode="keep_end",
 
        # precompute_ref_log_probs=True:
        # Runs the ref_model over all pairs ONCE before training starts,
        # saves log probs, then deletes ref_model from GPU memory.
        # This means during training only ONE model is in GPU memory (~6GB)
        # instead of two (~12GB), which is essential for the 23GB RTX 6000.
        # The cost is a slow precompute step — with 1000 pairs at max_length=4096
        # this takes ~20-30 minutes on GPU. With all 7785 pairs it takes ~7 hours.
        precompute_ref_log_probs=precompute_ref_log_probs,
 
        logging_steps=1,
        report_to="wandb",
        run_name=args.wandb_run_name,
 
        eval_strategy="no",
 
        bf16=True,
        fp16=False,
        use_cpu=args.cpu_test,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
 
        seed=24,
        disable_dropout=True,
        torch_empty_cache_steps=1,
        optim = "adamw_8bit",           # Saves ~2-3GB of VRAM
        dataloader_num_workers=1,  # Default is often 4+, which clones RAM usage per worker
        dataloader_pin_memory=False, # Reduces RAM overhead for large tensors
        save_strategy="steps",          # mejor que "epoch"
        save_steps=1,                 # guarda cada 100 steps
        save_total_limit=2,
        max_grad_norm = 2.0, # gradient clipping
        
    )
 
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[GPUMemoryWatchdog(check_interval=1)]
    )
    
    checkpoint = get_latest_checkpoint(args.checkpoint_dir)
    if checkpoint:
        print(f"🔁 Resuming from {checkpoint}")
    else:
        print("🆕 Starting fresh training")
 
    logger.info("Starting DPO training...")
    logger.info("(Precompute step runs first — ref_model runs once then gets deleted)")
    logger.info(f"(Estimated precompute time: ~{len(train_dataset) // 50} minutes on GPU)")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
 
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    # Free GPU memory before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    final_model_path = os.path.join(args.output_dir, f"final_model_{args.learning_rate}lr_{args.beta}beta")
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
 
    wandb.log({
        "train/final_loss":  train_result.metrics.get("train_loss", None),
        "train/total_steps": train_result.global_step,
    })
    wandb.finish()
 
    if args.cpu_test:
        logger.info("✅ CPU test passed!")
    else:
        logger.info("✅ DPO training complete!")
        logger.info(f"   Model saved to: {final_model_path}")
        logger.info(f"   Pairs trained on: {len(train_dataset)}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training for Multilingual Math Reasoning")
 
    parser.add_argument("--dataset_file",     type=str,   default=CONFIG["dataset_file"])
    parser.add_argument("--stats_file",       type=str,   default=CONFIG["stats_file"])
    parser.add_argument("--output_dir",       type=str,   default=CONFIG["output_dir"])
    parser.add_argument("--wandb_project",    type=str,   default=CONFIG["wandb_project"])
    parser.add_argument("--wandb_run_name",   type=str,   default=CONFIG["wandb_run_name"])
    parser.add_argument("--beta",             type=float, default=CONFIG["beta"])
    parser.add_argument("--learning_rate",    type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--num_epochs",       type=int,   default=CONFIG["num_epochs"])
    parser.add_argument("--batch_size",       type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--grad_accum_steps", type=int,   default=CONFIG["grad_accum_steps"])
    parser.add_argument("--max_length",       type=int,   default=CONFIG["max_length"])
    parser.add_argument("--max_pairs",        type=int,   default=CONFIG["max_pairs"],
                        help="Max pairs to use. 1000=~1hr, None=all pairs (~7hrs precompute)")
    parser.add_argument("--cpu_test",         action="store_true", default=CONFIG["cpu_test"])
    parser.add_argument("--checkpoint_dir",   type=str,   default=None)
    
    # parse_known_args silently ignores Colab's internal arguments
    args, _ = parser.parse_known_args()
    main(args)
