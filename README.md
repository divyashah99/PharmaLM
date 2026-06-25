# PharmaLM

Fine-tuning Phi-3-mini on pharmaceutical instruction data using LoRA and 4-bit quantization, with a built-in evaluation pipeline that scores outputs on ROUGE-L, BERTScore, and latency.

---

## What this is

PharmaLM fine-tunes Microsoft's Phi-3-mini-4k-instruct (3.8B) to answer clinical medication questions — dosage, drug interactions, contraindications, side effects — using supervised fine-tuning on the `sunny199/pharma-instruction-data` dataset.

The repo includes two things:

- `finetune_any_SLM.ipynb` — training pipeline with an OOP wrapper around Unsloth + TRL
- `evaluation_cells.py` + `test_cases_real.py` — post-training eval that runs without any extra infrastructure

---

## Stack

| Component | Library |
|---|---|
| Base model | `unsloth/Phi-3-mini-4k-instruct-bnb-4bit` |
| Fine-tuning | Unsloth + TRL (SFTTrainer) |
| PEFT method | LoRA |
| Quantization | 4-bit (bitsandbytes) |
| Eval metrics | ROUGE-L, BERTScore F1, p50/p90 latency |
| Runtime | Google Colab (T4 GPU) |

---

## Training config

```python
FineTuneConfig(
    model_name     = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    dataset_name   = "sunny199/pharma-instruction-data",
    epochs         = 1,
    lora_r         = 32,
    lora_alpha     = 32,
    lr             = 2e-5,
    per_device_bs  = 2,
    grad_acc_steps = 4,          # effective batch size = 8
    max_seq_length = 4096,
    warmup_ratio   = 0.1,
    optim          = "adamw_8bit",
    lora_targets   = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
```

4-bit quantization cuts VRAM usage by ~60% compared to full bf16, making it trainable on a free Colab T4.

---

## Project structure

```
PharmaLM/
├── finetune_any_SLM.ipynb   # training notebook
├── evaluation_cells.py      # SLMEvaluator class (ROUGE-L, BERTScore, latency)
├── test_cases_real.py       # 20 hardcoded pharma test cases + live dataset pull
├── eval_results.json        # sample eval output
└── README.md
```

---

## Quickstart

### 1. Install

```bash
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
pip install unsloth
pip install transformers==4.56.2 datasets==4.3.0
pip install trl==0.22.2
pip install rouge-score bert-score
```

### 2. Train

Open `finetune_any_SLM.ipynb` in Colab (GPU runtime required) and run all cells. The notebook handles model loading, dataset formatting, training, and saving LoRA adapters to `phi3_pharma_lora/`.

To swap the base model or dataset, edit the config at the bottom of the notebook:

```python
cfg = FineTuneConfig(
    model_name   = "unsloth/gemma-3-1b-it-bnb-4bit",  # swap model
    dataset_name = "tatsu-lab/alpaca",                  # swap dataset
)
```

Supported datasets out of the box: `tatsu-lab/alpaca`, `databricks/databricks-dolly-15k`, `anon8231489123/ShareGPT_Vicuna_unfiltered`, `OpenAssistant/oasst1`, `sunny199/pharma-instruction-data`. For anything else, the formatter falls back on column heuristics.

### 3. Evaluate

After training, paste the cells from `evaluation_cells.py` into the notebook and run:

```python
eval_cfg = EvalConfig(
    lora_path     = "phi3_pharma_lora",
    max_new_tokens = 300,
    run_rouge      = True,
    run_bertscore  = True,
    run_latency    = True,
    test_cases     = REAL_TEST_CASES,   # pulled live from dataset
)

evaluator = SLMEvaluator(eval_cfg)
report    = evaluator.evaluate()
evaluator.print_report(report)
evaluator.print_qualitative()
evaluator.save_results("eval_results.json")
```

The eval also supports comparing the fine-tuned model against the base model side by side to confirm the training actually helped.

---

## Eval results (sample run, 20 test cases)

| Metric | Score |
|---|---|
| ROUGE-L mean | see `eval_results.json` |
| BERTScore F1 mean | see `eval_results.json` |
| Inference latency p50 | ~16.3 s |
| Inference latency p90 | ~18.5 s |
| Correct clinical direction | 19 / 20 |

**Known issue:** the model continues generating past the answer on most responses (prompt template leaking). This is a stopping criterion problem, not a knowledge problem. Fix: set `eos_token_id` explicitly during generation or reduce `max_new_tokens`.

---

## Dataset

`sunny199/pharma-instruction-data` on HuggingFace Hub. Schema:

```
instruction  — role prompt ("Your role as a doctor...")
input        — patient question
output       — clinical answer
```

The eval test cases cover: dosage, side effects, drug interactions, drug class/mechanism, contraindications, pregnancy safety, storage, OTC counselling, overdose/toxicity, pharmacokinetics, and pharmacogenomics.

---

## Known limitations

- Trained for 1 epoch only. More epochs or a larger LoRA rank will likely improve output completeness.
- Not for clinical use. The model can and does make factual errors.
- Latency (~16s on T4) is too slow for real-time use. GGUF export + llama.cpp would bring this down significantly.

---

## License

Model weights follow the [Phi-3 license](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct). Dataset follows its original license on HuggingFace Hub. Code in this repo is MIT.
