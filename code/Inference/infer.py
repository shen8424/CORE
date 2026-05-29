"""
通用 vLLM 推理：运行时按顺序合并 stage1/stage2-pre/stage2-post/stage3 LoRA 到临时目录，
随后调用 swift infer 的 vLLM backend 推理，结束后自动删除临时融合模型。
"""
import argparse
import gc
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIBS = REPO_ROOT / 'libs'
LOCAL_SWIFT_PACKAGES = ('swift', 'qwen_vl_utils')
VLLM_AUXILIARY_WEIGHT_PREFIXES = (
    'logit_scale',
    'logit_bias',
    'vision2text.',
    'vision_proj.',
    'text_proj.',
)
sys.path.insert(0, str(LOCAL_LIBS))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import torch
import yaml
from peft import PeftModel
from swift.llm import get_model_tokenizer

BASE_MODEL = './Qwen2.5VL-3B'
DEFAULT_MODEL_TYPE = 'qwen2_5_vl'
DEFAULT_ATTN_IMPL = 'flash_attn'


def parse_args():
    parser = argparse.ArgumentParser(description='Generic vLLM runtime-merge inference')
    parser.add_argument('--config', help='YAML config path. If set, values are loaded from YAML first.')
    parser.add_argument('--base_model', default=None)
    parser.add_argument('--stage1_lora', default=None)
    parser.add_argument('--stage2_pre_lora', default=None)
    parser.add_argument('--stage2_post_lora', default=None)
    parser.add_argument('--stage3_lora', default=None)
    parser.add_argument('--pipeline_run_dir', default=None, help='stage1-2 pipeline run directory containing artifacts.yaml.')
    parser.add_argument('--stage3_run_dir', default=None, help='stage3 training output directory containing checkpoint-* and new_modules.pt.')
    parser.add_argument('--new_modules', default=None)
    parser.add_argument('--test_jsonl', default=None)
    parser.add_argument('--result_path', default=None)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--model_type', default=None)
    parser.add_argument('--torch_dtype', default=None)
    parser.add_argument('--attn_impl', default=None)
    parser.add_argument('--infer_backend', default=None)
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--repetition_penalty', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=None)
    parser.add_argument('--system', default=None)
    parser.add_argument('--write_batch_size', type=int, default=None)
    parser.add_argument('--temp_root', default=None, help='临时融合模型根目录，默认优先 /dev/shm。')
    parser.add_argument('--keep_temp_model', action='store_true', help='调试用：推理后不删除临时融合模型。')
    parser.add_argument('--skip_merge', action='store_true', help='兼容旧命令；当前脚本会忽略此参数。')
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def normalize_lora_checkpoints(raw_checkpoints) -> List[Dict[str, str]]:
    normalized = []
    for idx, item in enumerate(raw_checkpoints or []):
        if isinstance(item, dict):
            normalized.append({'name': item.get('name', f'lora_{idx}'), 'path': item.get('path', '')})
        else:
            normalized.append({'name': f'lora_{idx}', 'path': item})
    return normalized


def resolve_repo_path(path):
    if path is None or path == '':
        return path
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def latest_checkpoint(run_dir: str):
    run_dir = Path(resolve_repo_path(run_dir))
    final_checkpoint = run_dir / 'checkpoint-final'
    if final_checkpoint.is_dir():
        return str(final_checkpoint)
    checkpoints = []
    for candidate in run_dir.glob('checkpoint-*'):
        if not candidate.is_dir():
            continue
        suffix = candidate.name[len('checkpoint-'):]
        if suffix.isdigit():
            checkpoints.append((int(suffix), candidate))
    if checkpoints:
        checkpoints.sort(key=lambda item: item[0])
        return str(checkpoints[-1][1])
    return ''


def resolve_pipeline_artifacts_path(pipeline_run_dir: str):
    if not pipeline_run_dir:
        return ''
    path = Path(resolve_repo_path(pipeline_run_dir))
    if path.is_dir():
        path = path / 'artifacts.yaml'
    return str(path)


def load_pipeline_loras(pipeline_run_dir: str):
    artifacts_path = resolve_pipeline_artifacts_path(pipeline_run_dir)
    if not artifacts_path:
        return [], ''
    assert_path_exists(artifacts_path, 'pipeline artifacts.yaml')
    with open(artifacts_path, 'r', encoding='utf-8') as f:
        artifacts = yaml.safe_load(f) or {}
    stages = ['stage1', 'stage2-pre', 'stage2-post']
    missing = [stage for stage in stages if stage not in artifacts]
    if missing:
        raise KeyError(f'pipeline artifacts 缺少阶段 {missing}: {artifacts_path}')
    loras = [{'name': stage, 'path': artifacts[stage]['latest_checkpoint']} for stage in stages]
    new_modules = artifacts['stage2-post'].get('new_modules_pt', '')
    print(f'[CONFIG] loaded stage1/stage2-pre/stage2-post from {artifacts_path}', flush=True)
    return loras, new_modules


def resolve_stage3_outputs(stage3_run_dir: str):
    if not stage3_run_dir:
        return '', ''
    stage3_run_dir = resolve_repo_path(stage3_run_dir)
    assert_path_exists(stage3_run_dir, 'stage3_run_dir')
    checkpoint = latest_checkpoint(stage3_run_dir)
    if not checkpoint:
        raise FileNotFoundError(f'stage3_run_dir 下没有 checkpoint-final 或 checkpoint-*: {stage3_run_dir}')
    new_modules = os.path.join(stage3_run_dir, 'new_modules.pt')
    assert_path_exists(new_modules, 'stage3 new_modules.pt')
    return checkpoint, new_modules


def resolve_lora_paths(lora_checkpoints):
    resolved = []
    for item in lora_checkpoints:
        resolved.append({'name': item['name'], 'path': resolve_repo_path(item['path'])})
    return resolved


def resolve_config(args) -> Dict[str, Any]:
    config_path = resolve_repo_path(args.config) if args.config else ''
    cfg = load_yaml_config(config_path)
    model_cfg = cfg.get('model', {})
    infer_cfg = cfg.get('swift_infer_args', {})
    runtime_defaults = cfg.get('runtime_defaults', {})
    infer_defaults = cfg.get('infer_defaults', {})
    for key, value in runtime_defaults.items():
        model_cfg.setdefault(key, value)
    for key, value in infer_defaults.items():
        infer_cfg.setdefault(key, value)
    env_cfg = cfg.get('env', {})

    if args.gpu is not None:
        env_cfg['CUDA_VISIBLE_DEVICES'] = args.gpu
    if env_cfg.get('CUDA_VISIBLE_DEVICES') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(env_cfg['CUDA_VISIBLE_DEVICES'])

    pipeline_run_dir = args.pipeline_run_dir or model_cfg.get('pipeline_run_dir') or infer_cfg.get('pipeline_run_dir')
    stage3_run_dir = args.stage3_run_dir or model_cfg.get('stage3_run_dir') or infer_cfg.get('stage3_run_dir')
    stage3_lora, stage3_new_modules = resolve_stage3_outputs(stage3_run_dir)
    stage3_lora = args.stage3_lora or model_cfg.get('stage3_lora') or infer_cfg.get('stage3_lora') or stage3_lora

    pipeline_loras, pipeline_new_modules = load_pipeline_loras(pipeline_run_dir)
    lora_checkpoints = []
    if pipeline_loras:
        lora_checkpoints.extend(pipeline_loras)
        if stage3_lora:
            lora_checkpoints.append({'name': 'stage3', 'path': stage3_lora})
    else:
        raw_loras = model_cfg.get('lora_checkpoints') if 'lora_checkpoints' in model_cfg else infer_cfg.get('lora_checkpoints')
        if raw_loras is not None:
            lora_checkpoints = normalize_lora_checkpoints(raw_loras)
        else:
            for name, value in [
                ('stage1', args.stage1_lora or model_cfg.get('stage1_lora') or infer_cfg.get('stage1_lora')),
                ('stage2-pre', args.stage2_pre_lora or model_cfg.get('stage2_pre_lora') or infer_cfg.get('stage2_pre_lora')),
                ('stage2-post', args.stage2_post_lora or model_cfg.get('stage2_post_lora') or infer_cfg.get('stage2_post_lora')),
                ('stage3', stage3_lora),
            ]:
                if value:
                    lora_checkpoints.append({'name': name, 'path': value})
    lora_checkpoints = resolve_lora_paths(lora_checkpoints)

    new_modules = args.new_modules or model_cfg.get('new_modules') or infer_cfg.get('new_modules') or stage3_new_modules or pipeline_new_modules

    return {
        'base_model': resolve_repo_path(args.base_model or model_cfg.get('base_model') or infer_cfg.get('base_model') or BASE_MODEL),
        'model_type': args.model_type or infer_cfg.get('model_type') or DEFAULT_MODEL_TYPE,
        'attn_impl': args.attn_impl or infer_cfg.get('attn_impl') or DEFAULT_ATTN_IMPL,
        'torch_dtype': args.torch_dtype or infer_cfg.get('torch_dtype') or 'bfloat16',
        'infer_backend': args.infer_backend or infer_cfg.get('infer_backend') or 'vllm',
        'lora_checkpoints': lora_checkpoints,
        'new_modules': resolve_repo_path(new_modules),
        'test_jsonl': resolve_repo_path(args.test_jsonl or infer_cfg.get('val_dataset') or infer_cfg.get('test_jsonl')),
        'result_path': resolve_repo_path(args.result_path or infer_cfg.get('result_path')),
        'max_batch_size': args.max_batch_size or infer_cfg.get('max_batch_size', 6),
        'max_length': args.max_length or infer_cfg.get('max_length', 2048),
        'max_new_tokens': args.max_new_tokens or infer_cfg.get('max_new_tokens', 768),
        'temperature': args.temperature if args.temperature is not None else infer_cfg.get('temperature', 0),
        'top_p': args.top_p if args.top_p is not None else infer_cfg.get('top_p'),
        'top_k': args.top_k if args.top_k is not None else infer_cfg.get('top_k'),
        'repetition_penalty': args.repetition_penalty if args.repetition_penalty is not None else infer_cfg.get('repetition_penalty'),
        'num_beams': args.num_beams or infer_cfg.get('num_beams', 1),
        'system': args.system if args.system is not None else infer_cfg.get('system', ''),
        'write_batch_size': args.write_batch_size or infer_cfg.get('write_batch_size'),
        'temp_root': resolve_repo_path(args.temp_root or model_cfg.get('temp_root') or infer_cfg.get('temp_root')),
        'keep_temp_model': args.keep_temp_model or bool(model_cfg.get('keep_temp_model', False)),
    }


def torch_dtype_from_name(name: str):
    if name in {None, 'auto'}:
        return None
    if isinstance(name, torch.dtype):
        return name
    dtype_map = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    if name not in dtype_map:
        raise ValueError(f'Unsupported torch_dtype: {name}')
    return dtype_map[name]


def assert_path_exists(path: str, name: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'{name} 不存在: {path}')


def choose_temp_root(configured_root: str = None) -> str:
    if configured_root:
        os.makedirs(configured_root, exist_ok=True)
        return configured_root
    shm_root = '/dev/shm'
    if os.path.isdir(shm_root):
        free_bytes = shutil.disk_usage(shm_root).free
        if free_bytes >= 12 * 1024**3:
            return shm_root
        print(f'[WARN] /dev/shm 可用空间不足 ({free_bytes / 1024**3:.1f} GiB)，回退到 /tmp。', flush=True)
    fallback = '/tmp'
    os.makedirs(fallback, exist_ok=True)
    return fallback


def cleanup_stale_temp_models(temp_root: str):
    stale_dirs = sorted(glob.glob(os.path.join(temp_root, 'runtime_merged_*')))
    for stale_dir in stale_dirs:
        if os.path.isdir(stale_dir):
            shutil.rmtree(stale_dir, ignore_errors=True)
            print(f'[TEMP] removed stale merged model: {stale_dir}', flush=True)


def merge_lora_in_order(model, lora_checkpoints: List[Dict[str, str]]):
    if not lora_checkpoints:
        raise ValueError('lora_checkpoints is empty; configure pipeline_run_dir, stage3_run_dir, or explicit LoRA checkpoints before inference.')
    for item in lora_checkpoints:
        name = item['name']
        path = item['path']
        assert_path_exists(path, f'{name} LoRA checkpoint')
        print(f'[MERGE] {name}: {path}', flush=True)
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        cleanup_peft_metadata(model)
    return model


def cleanup_peft_metadata(model):
    """merge_and_unload 后清理 PEFT 残留标记，避免下一次 from_pretrained 误判为多 adapter。"""
    removed = []
    for attr in ('peft_config', 'active_adapter', 'active_adapters', 'modules_to_save'):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
                removed.append(attr)
            except AttributeError:
                pass
    if removed:
        print(f'[MERGE] cleaned PEFT metadata: {removed}', flush=True)


def normalize_new_module_key(name: str):
    for prefix in ('base_model.model.', 'model.'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace('.base_layer.', '.')
    if '.lora_A.' in name or '.lora_B.' in name:
        return None
    valid_keys = {
        'logit_scale',
        'logit_bias',
        'vision2text.in_proj_weight',
        'vision2text.in_proj_bias',
        'vision2text.out_proj.weight',
        'vision2text.out_proj.bias',
        'vision_proj.weight',
        'text_proj.weight',
    }
    return name if name in valid_keys else None


def load_new_modules(model, checkpoint_path: str):
    if not checkpoint_path:
        print('[LOAD] no new_modules checkpoint configured; using model initialization.', flush=True)
        return
    assert_path_exists(checkpoint_path, 'new_modules.pt')
    pt_state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    loaded = []
    for key, value in pt_state.items():
        normalized_key = normalize_new_module_key(key)
        target_key = normalized_key if normalized_key in model_state else key
        if target_key in model_state:
            model_state[target_key] = value.to(dtype=model_state[target_key].dtype)
            loaded.append(target_key)
    model.load_state_dict(model_state, strict=False)
    print(f'[LOAD] new_modules: {checkpoint_path} ({len(loaded)} params)', flush=True)
    if not loaded:
        raise RuntimeError(f'没有从 {checkpoint_path} 加载到任何 new_modules 参数')


def strip_vllm_auxiliary_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered = {}
    removed = []
    for key, value in state_dict.items():
        if key in VLLM_AUXILIARY_WEIGHT_PREFIXES or key.startswith(VLLM_AUXILIARY_WEIGHT_PREFIXES):
            removed.append(key)
            continue
        filtered[key] = value
    if removed:
        print(f'[SAVE] stripped vLLM-incompatible auxiliary weights: {removed}', flush=True)
    return filtered


def build_runtime_merged_model(cfg: Dict[str, Any], output_dir: str):
    print(f"[LOAD] base model: {cfg['base_model']}", flush=True)
    model, processor = get_model_tokenizer(
        cfg['base_model'],
        model_type=cfg['model_type'],
        torch_dtype=torch_dtype_from_name(cfg['torch_dtype']),
        device_map='cpu',
        attn_impl=cfg['attn_impl'],
    )
    model = merge_lora_in_order(model, cfg['lora_checkpoints'])
    load_new_modules(model, cfg['new_modules'])
    model.stage = None
    print(f'[SAVE] temporary merged model: {output_dir}', flush=True)
    model.save_pretrained(output_dir, state_dict=strip_vllm_auxiliary_weights(model.state_dict()), safe_serialization=True)
    processor.save_pretrained(output_dir)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_arg(cmd: List[str], name: str, value):
    if value is None:
        return
    cmd.extend([name, str(value)])


def make_swift_pythonpath_overlay() -> str:
    """Expose local swift packages without shadowing system transformers for vLLM."""
    overlay_dir = tempfile.mkdtemp(prefix='swift_pythonpath_')
    for package in LOCAL_SWIFT_PACKAGES:
        source = os.path.join(str(LOCAL_LIBS), package)
        target = os.path.join(overlay_dir, package)
        if not os.path.exists(source):
            raise FileNotFoundError(f'缺少本地 Python 包: {source}')
        os.symlink(source, target)
    return overlay_dir


def build_subprocess_pythonpath(prepend_path: str, current_pythonpath: str = '') -> str:
    local_libs_abs = os.path.abspath(str(LOCAL_LIBS))
    paths = [prepend_path]
    for path in (current_pythonpath or '').split(os.pathsep):
        if not path:
            continue
        if os.path.abspath(path) == local_libs_abs:
            continue
        paths.append(path)
    return os.pathsep.join(paths)


def run_swift_vllm_infer(cfg: Dict[str, Any], merged_model_dir: str):
    if cfg['infer_backend'] != 'vllm':
        raise ValueError(f"当前脚本用于 vLLM 推理，请设置 infer_backend: vllm，当前为: {cfg['infer_backend']}")
    os.makedirs(os.path.dirname(cfg['result_path']), exist_ok=True)
    cmd = [
        'swift', 'infer',
        '--model', merged_model_dir,
        '--val_dataset', cfg['test_jsonl'],
        '--result_path', cfg['result_path'],
        '--model_type', cfg['model_type'],
        '--infer_backend', 'vllm',
        '--torch_dtype', cfg['torch_dtype'],
        '--attn_impl', cfg['attn_impl'],
        '--max_length', str(cfg['max_length']),
        '--max_new_tokens', str(cfg['max_new_tokens']),
        '--temperature', str(cfg['temperature']),
        '--system', cfg['system'],
    ]
    append_arg(cmd, '--max_batch_size', cfg['max_batch_size'])
    append_arg(cmd, '--top_p', cfg['top_p'])
    append_arg(cmd, '--top_k', cfg['top_k'])
    append_arg(cmd, '--repetition_penalty', cfg['repetition_penalty'])
    append_arg(cmd, '--num_beams', cfg['num_beams'])
    append_arg(cmd, '--write_batch_size', cfg['write_batch_size'])

    overlay_dir = make_swift_pythonpath_overlay()
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = build_subprocess_pythonpath(overlay_dir, env.get('PYTHONPATH', ''))
        print(f'[ENV] swift infer uses local {LOCAL_SWIFT_PACKAGES}, system transformers for vLLM', flush=True)
        print('[INFER] ' + ' '.join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True, stdout=sys.stdout, stderr=sys.stderr)
    finally:
        shutil.rmtree(overlay_dir, ignore_errors=True)


def validate_config(cfg: Dict[str, Any]):
    assert_path_exists(cfg['base_model'], 'base_model')
    assert_path_exists(cfg['test_jsonl'], 'test_jsonl')
    if cfg.get('new_modules'):
        assert_path_exists(cfg['new_modules'], 'new_modules.pt')
    if not cfg.get('result_path'):
        raise ValueError('缺少 result_path')
    expected = ['stage1', 'stage2-pre', 'stage2-post', 'stage3']
    actual = [item['name'] for item in cfg['lora_checkpoints']]
    if not actual:
        raise ValueError('lora_checkpoints is empty; configure LoRA checkpoints before inference.')
    if actual != expected:
        print(f'[WARN] 当前 LoRA 顺序为 {actual}，建议严格使用 {expected}。', flush=True)
    for item in cfg['lora_checkpoints']:
        assert_path_exists(item['path'], f"{item['name']} LoRA checkpoint")


def main():
    cfg = resolve_config(parse_args())
    validate_config(cfg)
    temp_root = choose_temp_root(cfg.get('temp_root'))
    cleanup_stale_temp_models(temp_root)
    temp_dir = tempfile.mkdtemp(prefix='runtime_merged_', dir=temp_root)
    print(f'[TEMP] runtime merged model dir: {temp_dir}', flush=True)
    try:
        build_runtime_merged_model(cfg, temp_dir)
        run_swift_vllm_infer(cfg, temp_dir)
    finally:
        if cfg.get('keep_temp_model'):
            print(f'[TEMP] keep_temp_model=True，保留临时模型: {temp_dir}', flush=True)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f'[TEMP] removed: {temp_dir}', flush=True)


if __name__ == '__main__':
    main()
