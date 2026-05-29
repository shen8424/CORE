# CORE-Qwen

CORE-Qwen provides a staged Qwen2.5-VL training and inference pipeline.

The workflow is:

```text
stage1 -> stage2-pre -> stage2-post -> stage3 -> inference
```

- **Stage 1-2**: Trained on the [CAC](https://huggingface.co/datasets/SJJ0854/CAC) dataset for general forgery detection capabilities.
- **Stage 3**: Fine-tuned on a custom dataset (e.g., DGM4, MDSM(v1)) for domain-specific adaptation.

## Quick Start

### 1. Environment Setup

```bash
conda create -n CORE-Qwen python==3.10
conda activate CORE-Qwen
pip install ms-swift==3.10.3
pip install decord
pip install vllm==0.11.0
pip install deepspeed==0.17.6
pip install qwen_vl_utils==0.0.14
```

Download the pre-compiled flash-attention wheel from [Google Drive](https://drive.google.com/file/d/1b1Gxcwb3E7ft5vRyoJM60x5Di707_kvf/view?usp=sharing), then install it:

```bash
pip install /path/to/flash_attn-*.whl
```

### 2. Prepare Data and Base Model

Download the base model from [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and place it under:

```text
./Qwen2.5VL-3B
```

Download the dataset from [SJJ0854/CAC](https://huggingface.co/datasets/SJJ0854/CAC) and place the files under `./data`:

```bash
# Unzip images
unzip ./data/images.zip -d ./data/

# Ensure the following files are in ./data
# - stage1-5w.jsonl
# - stage2.jsonl
# - stage2-post.jsonl
```

### 3. Run Stage1/Stage2 Training

Run Stage1, Stage2-pre, and Stage2-post sequentially with one command:

```bash
cd /path/to/CORE-Qwen
python code/train_stage_pipeline.py --launcher deepspeed --num-gpus 2
```

Outputs are saved to:

```text
Outputs/pipeline_runs/<run_id>/
```


## Stage3 Training

Edit the Stage3 config:

```text
code/stage3/train_stage3.yaml
```

Set these fields:

```yaml
model_params:
  model_id_or_path: ./Qwen2.5VL-3B

path_params:
  pipeline_run_dir: /path/to/pipeline_run_dir

data_params:
  dataset:
    - /path/to/train.jsonl

output_params:
  output_root: ./Outputs/stage3/generic
```

Then launch Stage3:

```bash
cd /path/to/CORE-Qwen
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=0 NCCL_DEBUG=WARN \
  deepspeed --num_gpus 1 code/stage3/sft-stage3.py code/stage3/train_stage3.yaml
```

Stage3 outputs are saved by default to:

```text
Outputs/stage3/generic/<run_id>/
```

## Inference

Edit the inference config:

```text
code/Inference/infer.yaml
```

Set these fields:

```yaml
model:
  base_model: ./Qwen2.5VL-3B
  pipeline_run_dir: /path/to/pipeline_run_dir
  stage3_run_dir: /path/to/stage3_run_dir

swift_infer_args:
  val_dataset: /path/to/test.jsonl
  result_path: ./Outputs/inference/stage3_vllm_results.jsonl
```

Run inference:

```bash
cd /path/to/CORE-Qwen
bash code/Inference/run_inference.sh
```

Or run directly:

```bash
python code/Inference/infer.py --config code/Inference/infer.yaml
```

## Data Format

Training and inference JSONL files should follow the Swift multimodal format:

```json
{"images": ["/path/to/image.jpg"], "messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "Real"}], "fake_cls": "optional_class_name"}
```

`fake_cls` is optional for training and inference, but can be used by `code/Inference/evaluate.py` for grouped evaluation.
