"""
可选的持久化融合脚本。

默认推理请使用 infer.py；它会融合到临时目录并自动清理。
本脚本用于调试：读取 infer.yaml，同样支持 pipeline_run_dir + stage3_run_dir，
然后把融合后的模型保存到 --output。
"""
import argparse
from pathlib import Path

from infer import build_runtime_merged_model, resolve_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / 'code' / 'Inference' / 'infer.yaml'
DEFAULT_OUTPUT = REPO_ROOT / 'Outputs' / 'merged_models' / 'stage3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--pipeline_run_dir', default=None)
    parser.add_argument('--stage3_run_dir', default=None)
    args = parser.parse_args()

    infer_args = argparse.Namespace(
        config=args.config,
        base_model=None,
        stage1_lora=None,
        stage2_pre_lora=None,
        stage2_post_lora=None,
        stage3_lora=None,
        pipeline_run_dir=args.pipeline_run_dir,
        stage3_run_dir=args.stage3_run_dir,
        new_modules=None,
        test_jsonl=None,
        result_path=None,
        gpu=None,
        model_type=None,
        torch_dtype=None,
        attn_impl=None,
        infer_backend=None,
        max_batch_size=None,
        max_length=None,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        num_beams=None,
        system=None,
        write_batch_size=None,
        temp_root=None,
        keep_temp_model=False,
        skip_merge=False,
    )
    cfg = resolve_config(infer_args)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    build_runtime_merged_model(cfg, str(output))
    print(f'[INFO] merged model saved to {output}')


if __name__ == '__main__':
    main()
