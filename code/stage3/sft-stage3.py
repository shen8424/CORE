import sys
import os
import argparse
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIBS = REPO_ROOT / 'libs'
DEFAULT_CONFIG = Path(__file__).resolve().with_name('train_stage3.yaml')
sys.path.insert(0, str(LOCAL_LIBS))

import yaml
import shutil
from datetime import datetime
import torch
import torch.nn as nn

from swift.llm import (
    get_model_tokenizer, load_dataset, get_template,
    get_model_arch, get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, seed_everything
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.tuners import Swift, LoraConfig
from swift.plugin import MeanMetric
from peft import PeftModel

logger = get_logger()


def resolve_repo_path(path):
    if path is None or path == '':
        return path
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def resolve_repo_paths(paths):
    if isinstance(paths, str):
        return [resolve_repo_path(paths)]
    return [resolve_repo_path(path) for path in paths]


def resolve_pipeline_artifacts_path(path_cfg):
    pipeline_run_dir = path_cfg.get('pipeline_run_dir')
    pipeline_artifacts = path_cfg.get('pipeline_artifacts')
    value = pipeline_run_dir or pipeline_artifacts
    if value in (None, '', False):
        return None

    path = Path(resolve_repo_path(value))
    if path.is_dir():
        path = path / 'artifacts.yaml'
    return path


def apply_pipeline_artifacts(path_cfg):
    artifacts_path = resolve_pipeline_artifacts_path(path_cfg)
    if artifacts_path is None:
        return
    if not artifacts_path.exists():
        raise FileNotFoundError(f"pipeline artifacts 不存在: {artifacts_path}")
    with open(artifacts_path, 'r') as f:
        artifacts = yaml.safe_load(f) or {}

    required_stages = ['stage1', 'stage2-pre', 'stage2-post']
    missing = [stage for stage in required_stages if stage not in artifacts]
    if missing:
        raise KeyError(f"pipeline_artifacts 缺少阶段 {missing}: {artifacts_path}")

    path_cfg['lora_checkpoints'] = [
        {'name': stage, 'path': artifacts[stage]['latest_checkpoint']}
        for stage in required_stages
    ]
    path_cfg['new_modules_checkpoint'] = artifacts['stage2-post']['new_modules_pt']
    logger.info(f"Loaded stage1/stage2-pre/stage2-post artifacts from: {artifacts_path}")


def normalize_config_paths(config):
    model_cfg = config['model_params']
    path_cfg = config['path_params']
    output_cfg = config.setdefault('output_params', {})
    data_cfg = config['data_params']
    data_defaults = config.get('data_defaults', {})
    for key, value in data_defaults.items():
        data_cfg.setdefault(key, value)
    training_cfg = config['training_params']

    apply_pipeline_artifacts(path_cfg)

    model_cfg['model_id_or_path'] = resolve_repo_path(model_cfg['model_id_or_path'])
    output_root = output_cfg.get('output_root')
    if not output_root:
        raise KeyError('Missing output_params.output_root')
    output_cfg['output_root'] = resolve_repo_path(output_root)
    path_cfg['new_modules_checkpoint'] = resolve_repo_path(path_cfg.get('new_modules_checkpoint', ''))
    if 'deepspeed' in training_cfg and training_cfg['deepspeed']:
        training_cfg['deepspeed'] = resolve_repo_path(training_cfg['deepspeed'])
    data_cfg['dataset'] = resolve_repo_paths(data_cfg['dataset'])

    normalized_loras = []
    for idx, item in enumerate(path_cfg.get('lora_checkpoints') or []):
        if isinstance(item, dict):
            normalized = dict(item)
            normalized['path'] = resolve_repo_path(normalized.get('path', ''))
        else:
            normalized = {'name': f'lora_{idx}', 'path': resolve_repo_path(item)}
        normalized_loras.append(normalized)
    path_cfg['lora_checkpoints'] = normalized_loras


def validate_stage3_inputs(config):
    errors = []
    model_path = Path(config['model_params']['model_id_or_path'])
    if not model_path.exists():
        errors.append(f"model_id_or_path 不存在: {model_path}")
    elif not any(model_path.glob('*.safetensors')) and not any(model_path.glob('*.bin')):
        errors.append(f"model_id_or_path 中没有模型权重: {model_path}")

    for item in config['path_params'].get('lora_checkpoints') or []:
        checkpoint = Path(item.get('path', ''))
        if not checkpoint.exists():
            errors.append(f"{item.get('name', 'LoRA')} checkpoint 不存在: {checkpoint}")
    if not config['path_params'].get('lora_checkpoints'):
        errors.append('lora_checkpoints 为空：Stage3 必须基于 stage1/stage2 pipeline artifacts 或显式 LoRA checkpoints 继续训练。')
    new_modules = config['path_params'].get('new_modules_checkpoint')
    if new_modules and not Path(new_modules).exists():
        errors.append(f"new_modules_checkpoint 不存在: {new_modules}")
    for dataset in config['data_params'].get('dataset', []):
        if not Path(dataset).exists():
            errors.append(f"dataset 不存在: {dataset}")
    deepspeed_config = config['training_params'].get('deepspeed')
    if deepspeed_config and not Path(deepspeed_config).exists():
        errors.append(f"deepspeed config 不存在: {deepspeed_config}")

    if errors:
        raise FileNotFoundError('Stage3 preflight check failed:\n  - ' + '\n  - '.join(errors))


def configure_distributed_cuda_device():
    """Bind each DeepSpeed rank to its local GPU before any CUDA model load."""
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('LOCAL_RANK_ID', '-1')))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Set CUDA device to local_rank={local_rank}")
    return local_rank


NEW_MODULE_NAMES = ["vision2text", "logit_scale", "logit_bias"]


def load_new_modules_from_checkpoint(model, checkpoint_path):
    """从 checkpoint 加载新模块权重"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"未找到新模块权重文件: {checkpoint_path}，新模块将使用随机初始化")
        return

    pt_state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    loaded = []
    for k, v in pt_state.items():
        if k in model_state:
            model_state[k] = v
            loaded.append(k)
    model.load_state_dict(model_state, strict=False)
    logger.info(f"从 {checkpoint_path} 加载了 {len(loaded)} 个新模块权重")


def get_ordered_lora_checkpoints(path_cfg):
    """读取需要按顺序合并的 LoRA checkpoints。"""
    lora_checkpoints = path_cfg.get('lora_checkpoints')
    if lora_checkpoints is not None:
        ordered_checkpoints = []
        for idx, item in enumerate(lora_checkpoints):
            if isinstance(item, dict):
                name = item.get('name', f'lora_{idx}')
                checkpoint = item.get('path', '')
            else:
                name = f'lora_{idx}'
                checkpoint = item
            ordered_checkpoints.append((name, checkpoint))
        return ordered_checkpoints

    legacy_checkpoints = [
        ('stage2-pre', path_cfg.get('lora_checkpoint_stage2_pre', '')),
        ('stage2-post', path_cfg.get('lora_checkpoint_stage2_post', '')),
    ]
    return [(name, checkpoint) for name, checkpoint in legacy_checkpoints if checkpoint]


def merge_lora_checkpoints_in_order(model, ordered_checkpoints):
    """基于当前模型按给定顺序逐个 merge LoRA。"""
    if not ordered_checkpoints:
        raise ValueError('Stage3 requires lora_checkpoints.')

    for name, checkpoint in ordered_checkpoints:
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{name} LoRA checkpoint 不存在: {checkpoint}")
        logger.info(f"Merging {name} LoRA checkpoint from: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        cleanup_peft_metadata(model)
        logger.info(f"已合并 {name} LoRA checkpoint: {checkpoint}")
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
        logger.info(f"Cleaned PEFT metadata after merge: {removed}")


class DetailLossTrainer(Seq2SeqTrainer):
    """自定义 Trainer：自动识别并记录各子 loss"""

    @staticmethod
    def _find_inner_model(model):
        """穿透 Swift/PEFT 包装，找到实际执行 forward 的模型"""
        if hasattr(model, 'base_model') and hasattr(model.base_model, '_log_loss_gen'):
            return model.base_model
        if hasattr(model, 'model') and hasattr(model.model, 'base_model'):
            candidate = model.model.base_model
            if hasattr(candidate, '_log_loss_gen'):
                return candidate
        if hasattr(model, '_log_loss_gen'):
            return model
        return model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        actual_model = self.accelerator.unwrap_model(model)
        inner_model = self._find_inner_model(actual_model)
        for attr, metric_key in [
            ('_log_loss_gen', 'loss_gen'),
            ('_log_loss_cl',   'loss_cl'),
            ('_log_loss_box',  'loss_box'),
        ]:
            val = getattr(inner_model, attr, None)
            if val is not None:
                if metric_key not in self._custom_metrics:
                    self._custom_metrics[metric_key] = MeanMetric(nan_value=None)
                self._custom_metrics[metric_key].update(val.to(loss.device))

        return loss


def main(config_path: str):
    # --- 1. 加载配置 ---
    config_path = resolve_repo_path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    normalize_config_paths(config)
    validate_stage3_inputs(config)

    run_cfg = config['run_params']
    model_cfg = config['model_params']
    path_cfg = config['path_params']
    output_cfg = config['output_params']
    data_cfg = config['data_params']
    lora_cfg = config['lora_params']
    training_cfg = config['training_params']

    # --- 2. 初始化 ---
    local_rank = configure_distributed_cuda_device()
    seed_everything(run_cfg['seed'])
    stage = run_cfg['stage']
    model_id_or_path = model_cfg['model_id_or_path']

    system_prompts = config.get('system_prompts') or model_cfg.get('system_prompts', {})
    system = system_prompts.get(stage, '')
    logger.info(f"Stage: {stage}, System Prompt: '{system}'")

    current_time = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = os.path.join(output_cfg['output_root'], current_time)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Output directory: {output_dir}')

    # --- 3. 加载 base 模型 ---
    model_kwargs = {}
    if local_rank >= 0:
        model_kwargs['device_map'] = f'cuda:{local_rank}'
    model, processor = get_model_tokenizer(
        model_id_or_path,
        model_type="qwen2_5_vl",
        attn_impl='flash_attn',
        **model_kwargs,
    )
    model.stage = stage

    # --- 4. 基于 base model 按顺序合并 stage1 -> stage2-pre -> stage2-post 的 LoRA ---
    model = merge_lora_checkpoints_in_order(model, get_ordered_lora_checkpoints(path_cfg))
    model.stage = stage

    # --- 5. 加载 stage2-post 新模块权重 ---
    new_modules_ckpt = path_cfg.get('new_modules_checkpoint', '')
    load_new_modules_from_checkpoint(model, new_modules_ckpt)

    # --- 6. 为下游微调创建新的 LoRA ---
    target_modules = get_multimodal_target_regex(
        model,
        freeze_llm=lora_cfg['freeze_llm'],
        freeze_vit=lora_cfg['freeze_vit'],
        freeze_aligner=lora_cfg['freeze_aligner']
    )
    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=lora_cfg['lora_rank'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=target_modules
    )
    model = Swift.prepare_model(model, lora_config)

    # --- 7. 确保新模块保持可训练 ---
    for name, param in model.named_parameters():
        if any(m in name for m in NEW_MODULE_NAMES):
            param.requires_grad = True

    logger.info(f'LoRA Config: {lora_config}')
    logger.info(f'Trainable Parameters: {get_model_parameter_info(model)}')

    # --- 8. 配置模板 ---
    template = get_template(
        model.model_meta.template,
        processor,
        default_system=system,
        max_length=data_cfg['max_length'],
        remove_unused_columns=False
    )
    template.set_mode('train')
    template.set_stage(stage)
    if template.use_model:
        template.model = model

    # --- 9. 加载数据集 ---
    train_dataset, _ = load_dataset(
        data_cfg['dataset'],
        split_dataset_ratio=data_cfg['split_dataset_ratio'],
        num_proc=data_cfg['num_proc'],
        seed=run_cfg['seed']
    )
    train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=run_cfg['seed'])
    logger.info(f'Train dataset loaded: {train_dataset}')

    # --- 10. 训练参数 ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        data_seed=run_cfg['seed'],
        **training_cfg
    )

    # --- 11. 训练 ---
    trainer = DetailLossTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        template=template,
    )
    trainer.train()

    # --- 12. 保存新模块权重 ---
    inner_model = getattr(model, 'model', model)
    new_modules_state_dict = {}
    for name, param in inner_model.named_parameters():
        if any(m in name for m in NEW_MODULE_NAMES):
            new_modules_state_dict[name] = param.data.clone()

    save_path = os.path.join(output_dir, "new_modules.pt")
    torch.save(new_modules_state_dict, save_path)
    logger.info(f"New module parameters saved to {save_path}")

    # --- 13. 保存配置 ---
    final_config_path = os.path.join(output_dir, 'config.yaml')
    shutil.copy(config_path, final_config_path)
    logger.info(f"Configuration saved to {final_config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        nargs='?',
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument('--local_rank', type=int, default=-1)
    args, _ = parser.parse_known_args()
    if args.local_rank >= 0:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    main(args.config_path)
