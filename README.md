# LLM Safety Under Distribution Shift

Coursework #3

## Description

Experimental study of how distribution shift affects the ability of open chat models to refuse harmful requests. This work evaluates the degradation of safety mechanisms under changes in input style, pragmatics, and surface form.

## Research Question and Hypotheses

**Primary question:** How does Attack Success Rate (ASR) change when adversarial prompts from the StrongReject benchmark are combined with different types of distribution shift?

**Hypotheses:**
- H1: Semantic and pragmatic shifts (indirect framing, academic style) are more effective at bypassing safety mechanisms than surface-level distortions (typos, symbol noise).
- H2: Safety alignment implemented via prompt engineering and post-training is highly sensitive to changes in request framing.

## Methodology

| Component | Description |
|-----------|-------------|
| **Model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Quantization** | 4-bit NF4 (`bitsandbytes`), `torch.float16` |
| **Environment** | Google Colab (GPU T4, ~14.5 GB VRAM) |
| **Dataset** | `walledai/StrongREJECT` (313 examples, 6 harm categories) |
| **Sample** | Stratified, n=30 (5 examples per category) |
| **Shift types** | `none` (baseline), `typos`, `indirect`, `formal`, `noise` |
| **Metric** | Attack Success Rate (ASR) = 1 - Refusal Rate |
| **Refusal detection** | Heuristic keyword matching in model responses |

### Shift Types

- `none`: Original prompt from benchmark
- `typos`: Random character swaps, deletions, insertions (8% rate)
- `indirect`: Wrapped in hypothetical/research framing
- `formal`: Prefixed with instruction to respond in academic style
- `noise`: Random symbols inserted between characters (5% rate)

## Results

Aggregate metrics by shift type (Qwen2.5-3B, n=30):

| Shift | ASR | Refusal Rate | Interpretation |
|-------|-----|--------------|----------------|
| `indirect` | 0.800 | 0.200 | Indirect framing most effective at bypassing safety |
| `formal` | 0.733 | 0.267 | Academic style reduces refusal rate by ~27 pp vs baseline |
| `none` | 0.467 | 0.533 | Baseline model performance without shifts |
| `typos` | 0.467 | 0.533 | Orthographic distortions do not affect refusal decisions |
| `noise` | 0.433 | 0.567 | Symbol noise slightly improves robustness |

## Key Findings

1. **Differential vulnerability:** Semantic shifts (`indirect`, `formal`) increase ASR by 26-33 percentage points relative to baseline, while surface-level distortions (`typos`, `noise`) have negligible effect.

2. **Pragmatic framing effect:** Instructions to respond in formal academic style shift the model from "safe assistant" mode to "expert analyst" mode, lowering the refusal threshold.

3. **Robustness to orthographic noise:** The model correctly extracts harmful intent even with typos and random symbols, indicating reliable tokenization and semantic understanding, but not robust safety mechanisms.

4. **Practical implication:** Safety implemented solely via alignment and system prompts is vulnerable to contextual and stylistic manipulation. Production systems require multi-layered controls (external classifiers, intent filtering, post-generation validation).

## Limitations

- **Sample size:** n=30 demonstrates trends but has wide confidence intervals (~±18%). Statistically significant conclusions require n≥100.
- **Heuristic refusal detection:** The `is_refusal()` function uses keyword matching and may produce false positives/negatives. LLM-as-a-judge or manual audit of a subset is preferred for rigorous evaluation.
- **Single architecture:** Results apply to Qwen2.5 family. Other architectures (Mistral, Llama, Gemma) may exhibit different sensitivity to shifts.
- **Execution environment:** Experiments conducted in Google Colab with 4-bit quantization, which may slightly affect generation quality compared to fp16/bf16 inference.

## Reproduction

### Requirements
- Python 3.10+
- Google Colab with GPU access (T4 or higher recommended)
- Hugging Face token with access to `walledai/StrongREJECT`

### Installation
```bash
pip install -q accelerate bitsandbytes transformers datasets pandas seaborn matplotlib tqdm huggingface_hub>=0.20.0
```

### References
- StrongREJECT Benchmark. https://huggingface.co/datasets/walledai/StrongREJECT
- Qwen2.5-3B-Instruct. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
- Wolf et al. (2020). HuggingFace Transformers.
