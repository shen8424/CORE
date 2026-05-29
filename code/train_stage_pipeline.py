import argparse
import copy
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LIBS = REPO_ROOT / 'libs'
TRAIN_SCRIPT = REPO_ROOT / 'code' / 'sft-stage1or2.py'
DEFAULT_CONFIG = REPO_ROOT / 'code' / 'train_stage1or2.yaml'
DEFAULT_STAGES = ['stage1', 'stage2-pre', 'stage2-post']


def resolve_repo_path(path):
    if not path:
        return path
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_repo_paths(paths):
    if isinstance(paths, str):
        return [resolve_repo_path(paths)]
    return [resolve_repo_path(path) for path in paths]


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def prepend_env_path(env, name, path):
    path = str(path)
    existing = env.get(name, '')
    parts = [item for item in existing.split(os.pathsep) if item]
    parts = [item for item in parts if Path(item).resolve() != Path(path).resolve()]
    env[name] = os.pathsep.join([path] + parts)


def build_subprocess_env():
    env = os.environ.copy()
    prepend_env_path(env, 'PYTHONPATH', LOCAL_LIBS)
    env['CORE_QWEN_REPO_ROOT'] = str(REPO_ROOT)
    env['CORE_QWEN_LOCAL_LIBS'] = str(LOCAL_LIBS)
    return env


def latest_stage_output(stage_root):
    if not stage_root.exists():
        return None
    dirs = [path for path in stage_root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def latest_checkpoint(output_dir):
    final_checkpoint = output_dir / 'checkpoint-final'
    if final_checkpoint.is_dir():
        return final_checkpoint
    checkpoints = []
    for path in output_dir.glob('checkpoint-*'):
        if not path.is_dir():
            continue
        suffix = path.name[len('checkpoint-'):]
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def dry_run_artifacts(stage_outputs_root, stage):
    output_dir = stage_outputs_root / stage / 'DRY_RUN'
    return {
        'stage': stage,
        'output_dir': str(output_dir),
        'new_modules_pt': str(output_dir / 'new_modules.pt'),
        'latest_checkpoint': str(output_dir / 'checkpoint-final'),
    }


def read_stage_artifacts(stage_outputs_root, stage):
    output_dir = latest_stage_output(stage_outputs_root / stage)
    if output_dir is None:
        raise RuntimeError(f'No output directory found for {stage} under {stage_outputs_root / stage}')

    artifacts_path = output_dir / 'artifacts.yaml'
    artifacts = {}
    if artifacts_path.exists():
        artifacts = load_yaml(artifacts_path) or {}

    new_modules_pt = artifacts.get('new_modules_pt') or str(output_dir / 'new_modules.pt')
    checkpoint = artifacts.get('latest_checkpoint') or latest_checkpoint(output_dir)

    new_modules_pt = Path(new_modules_pt)
    checkpoint = Path(checkpoint) if checkpoint else None

    if not new_modules_pt.exists():
        raise RuntimeError(f'{stage} finished but new_modules.pt was not found: {new_modules_pt}')
    if checkpoint is None or not checkpoint.exists():
        raise RuntimeError(
            f'{stage} finished but no LoRA checkpoint was found under {output_dir}. '
            'The pipeline enables save_final_checkpoint automatically; check the training log for save failures.'
        )

    return {
        'stage': stage,
        'output_dir': str(output_dir),
        'new_modules_pt': str(new_modules_pt),
        'latest_checkpoint': str(checkpoint),
    }


def validate_model_dir(stage, model_path, errors):
    if not model_path:
        return
    if not model_path.exists():
        errors.append(f'{stage} model_id_or_path not found: {model_path}')
        return
    if not model_path.is_dir():
        errors.append(f'{stage} model_id_or_path is not a directory: {model_path}')
        return

    required_files = ['config.json']
    missing_files = [name for name in required_files if not (model_path / name).exists()]
    has_weights = any(model_path.glob('*.safetensors')) or any(model_path.glob('*.bin'))
    if missing_files:
        errors.append(f'{stage} model directory is missing files {missing_files}: {model_path}')
    if not has_weights:
        errors.append(f'{stage} model directory has no *.safetensors or *.bin weights: {model_path}')


def validate_base_inputs(config, stages):
    errors = []
    if not LOCAL_LIBS.is_dir():
        errors.append(f'local libs directory not found: {LOCAL_LIBS}')
    for required_lib in ['swift', 'transformers']:
        if not (LOCAL_LIBS / required_lib).exists():
            errors.append(f'local {required_lib} package not found under libs: {LOCAL_LIBS / required_lib}')

    deepspeed_config = resolve_repo_path(config.get('deepspeed_config'))
    if deepspeed_config and not deepspeed_config.exists():
        errors.append(f'deepspeed_config not found: {deepspeed_config}')

    checked_models = set()
    for stage in stages:
        stage_cfg = config['stages'][stage]
        model_path = resolve_repo_path(stage_cfg.get('model_id_or_path'))
        if model_path not in checked_models:
            validate_model_dir(stage, model_path, errors)
            checked_models.add(model_path)
        for dataset_path in resolve_repo_paths(stage_cfg.get('dataset', [])):
            if not dataset_path.exists():
                errors.append(f'{stage} dataset not found: {dataset_path}')

    if errors:
        joined = '\n  - '.join(errors)
        raise FileNotFoundError(f'Preflight check failed:\n  - {joined}')


def build_stage_config(base_config, stage, stage_outputs_root, previous_artifacts):
    config = copy.deepcopy(base_config)
    config.setdefault('run_params', {})['stage'] = stage
    config.setdefault('path_params', {})['base_dir'] = str(stage_outputs_root)

    stage_cfg = config['stages'][stage]
    stage_cfg['save_final_checkpoint'] = True

    if stage == 'stage2-pre':
        stage1 = previous_artifacts['stage1']
        stage_cfg['new_modules_pt'] = stage1['new_modules_pt']
        stage_cfg['lora_checkpoints'] = [stage1['latest_checkpoint']]
    elif stage == 'stage2-post':
        stage1 = previous_artifacts['stage1']
        stage2_pre = previous_artifacts['stage2-pre']
        stage_cfg['new_modules_pt'] = stage2_pre['new_modules_pt']
        stage_cfg['lora_checkpoints'] = [stage1['latest_checkpoint'], stage2_pre['latest_checkpoint']]

    return config


def build_command(args, stage_config_path):
    if args.launcher == 'python':
        return [args.python, str(TRAIN_SCRIPT), str(stage_config_path)]

    if args.launcher == 'deepspeed':
        command = ['deepspeed']
        if args.num_gpus is not None:
            command += ['--num_gpus', str(args.num_gpus)]
        if args.master_port is not None:
            command += ['--master_port', str(args.master_port)]
        for launcher_arg in args.launcher_arg:
            command += shlex.split(launcher_arg)
        command += [str(TRAIN_SCRIPT), str(stage_config_path)]
        return command

    raise ValueError(f'Unsupported launcher: {args.launcher}')


def run_stage(args, stage, stage_config_path):
    command = build_command(args, stage_config_path)
    env = build_subprocess_env()
    print(f'\n========== Running {stage} ==========')
    print('Command:', ' '.join(shlex.quote(part) for part in command), flush=True)
    print(f'Using local libs: {LOCAL_LIBS}', flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run stage1 -> stage2-pre -> stage2-post sequentially.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG), help='Base YAML config path.')
    parser.add_argument('--pipeline-dir', default='', help='Directory for generated configs, outputs, and summary.')
    parser.add_argument('--stages', nargs='+', default=DEFAULT_STAGES, help='Stages to run in order.')
    parser.add_argument('--launcher', choices=['python', 'deepspeed'], default='deepspeed')
    parser.add_argument('--python', default=sys.executable, help='Python executable used by --launcher python.')
    parser.add_argument('--num-gpus', type=int, default=2, help='Passed to deepspeed as --num_gpus.')
    parser.add_argument('--master-port', type=int, default=29549, help='Passed to deepspeed as --master_port.')
    parser.add_argument('--launcher-arg', action='append', default=[], help='Extra launcher args; can be repeated.')
    parser.add_argument('--skip-preflight', action='store_true', help='Skip model/dataset/deepspeed path checks.')
    parser.add_argument('--dry-run', action='store_true', help='Write generated configs and print commands only.')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    base_config = load_yaml(config_path)

    unknown_stages = [stage for stage in args.stages if stage not in base_config.get('stages', {})]
    if unknown_stages:
        raise ValueError(f'Unknown stages in --stages: {unknown_stages}. Available: {list(base_config.get("stages", {}))}')

    expected_order = DEFAULT_STAGES
    if args.stages != expected_order:
        print(f'Warning: custom stage order {args.stages}; default dependency order is {expected_order}', flush=True)

    if not args.skip_preflight:
        validate_base_inputs(base_config, args.stages)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_dir = resolve_repo_path(args.pipeline_dir) if args.pipeline_dir else REPO_ROOT / 'Outputs' / 'pipeline_runs' / run_id
    config_dir = pipeline_dir / 'configs'
    stage_outputs_root = pipeline_dir / 'stage_outputs'
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    stage_outputs_root.mkdir(parents=True, exist_ok=True)

    print(f'Pipeline directory: {pipeline_dir}')
    print(f'Stage outputs root: {stage_outputs_root}')
    print(f'Local libs injected into PYTHONPATH: {LOCAL_LIBS}')

    previous_artifacts = {}
    for stage in args.stages:
        stage_config = build_stage_config(base_config, stage, stage_outputs_root, previous_artifacts)
        stage_config_path = config_dir / f'{stage}.yaml'
        dump_yaml(stage_config, stage_config_path)
        run_stage(args, stage, stage_config_path)
        if args.dry_run:
            previous_artifacts[stage] = dry_run_artifacts(stage_outputs_root, stage)
            continue

        previous_artifacts[stage] = read_stage_artifacts(stage_outputs_root, stage)
        summary_path = pipeline_dir / 'artifacts.yaml'
        dump_yaml(previous_artifacts, summary_path)
        print(f'{stage} artifacts: {previous_artifacts[stage]}', flush=True)

    if args.dry_run:
        print('Dry run completed. Generated configs only; no training was launched.')
    else:
        print(f'Pipeline completed. Summary: {pipeline_dir / "artifacts.yaml"}')


if __name__ == '__main__':
    main()
