import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "XueZhang-bjtu/1.5B-cold-start-SFT"

# The prompt already ends with "<think>\n" (assistant turn opener).
# chosen/rejected also start with "<think>\n", so naive concatenation
# produces a duplicate token. Strip it from the response before joining.
THINK_PREFIX = "<think>\n"


def get_batch_logps(logits, labels, average_log_probs=False):
    """
    Standard DPO log-prob calculation matching TRL's internal logic.
    Only tokens where labels != -100 contribute to the sum.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch, seq, vocab) and labels (batch, seq) must match.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0  # dummy index for masked positions

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_probs:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def main():
    parser = argparse.ArgumentParser(description="Precompute DPO reference log-probs")
    parser.add_argument("--input_file",   type=str, required=True,
                        help="Path to the .jsonl dataset file")
    parser.add_argument("--output_dir",   type=str, default="./dpo_dataset_precomputed",
                        help="Directory to save the precomputed dataset")
    parser.add_argument("--max_length",   type=int, default=8000,
                        help="Max token length; must match max_length used in DPO training")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Limit total examples processed (e.g. 100 for a quick test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use fp16 on pre-Ampere GPUs (no native bf16); bf16 silently produces garbage there
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading model on {device} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {args.input_file}")
    ds = load_dataset("json", data_files=args.input_file)["train"]
    if args.num_examples is not None:
        ds = ds.select(range(min(args.num_examples, len(ds))))
    print(f"Processing {len(ds)} pairs...")

    def process_fn(example):
        # add_special_tokens=False: BOS is already embedded in the prompt string
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        results = {}
        with torch.no_grad():
            for key, response in [("chosen", example["chosen"]), ("rejected", example["rejected"])]:
                # Strip duplicate <think>\n already present at end of prompt
                response_body = (
                    response[len(THINK_PREFIX):]
                    if response.startswith(THINK_PREFIX)
                    else response
                )
                full_text = example["prompt"] + response_body

                enc = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                labels = enc["input_ids"].clone()
                labels[0, :prompt_len] = -100

                outputs = model(**enc)
                logps = get_batch_logps(outputs.logits, labels)
                results[f"reference_{key}_logps"] = logps.item()
                del outputs, logps
                if device == "cuda":
                    torch.cuda.empty_cache()

        return results

    processed = ds.map(process_fn, batched=False, desc="Precomputing log-probs")

    os.makedirs(args.output_dir, exist_ok=True)
    processed.save_to_disk(args.output_dir)
    print(f"\nDone! Precomputed dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
