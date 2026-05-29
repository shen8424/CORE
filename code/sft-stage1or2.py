import os
import sys
import yaml
import argparse
import shutil
from datetime import datetime
from functools import partial
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOCAL_LIBS = os.path.join(REPO_ROOT, 'libs')
sys.path.insert(0, LOCAL_LIBS)

from swift.llm import (
    get_model_tokenizer, load_dataset, get_template,
    get_model_arch, get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.plugin import MeanMetric
from peft import PeftModel

logger = get_logger()

# 基础模型路径（stage2-pre 回退用）
BASE_MODEL_PATH = os.path.join(REPO_ROOT, 'Qwen2.5VL-3B')


def resolve_repo_path(path):
    if not path:
        return path
    path = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def resolve_repo_paths(paths):
    if isinstance(paths, str):
        return resolve_repo_path(paths)
    return [resolve_repo_path(path) for path in paths]


def configure_distributed_cuda_device():
    """Bind each DeepSpeed rank to its local GPU before any CUDA model load."""
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('LOCAL_RANK_ID', '-1')))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Set CUDA device to local_rank={local_rank}")
    return local_rank


def get_lora_checkpoints(stage_cfg):
    lora_checkpoints = stage_cfg.get('lora_checkpoints')
    if lora_checkpoints is None:
        lora_checkpoint = stage_cfg.get('lora_checkpoint', '')
        lora_checkpoints = [lora_checkpoint] if lora_checkpoint else []
    elif isinstance(lora_checkpoints, str):
        lora_checkpoints = [lora_checkpoints]
    return [ckpt for ckpt in lora_checkpoints if ckpt]


def cleanup_peft_metadata(model):
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


def merge_lora_checkpoints(model, checkpoints, stage):
    if not checkpoints:
        logger.warning(f"未配置 LoRA checkpoint，将只基于当前 base/new_modules 创建 {stage} LoRA")
        return model

    for checkpoint in checkpoints:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"LoRA checkpoint 不存在: {checkpoint}")
        logger.info(f"Loading and merging LoRA checkpoint from: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        cleanup_peft_metadata(model)
    model.stage = stage
    return model


def normalize_new_module_key(name):
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


def collect_new_modules_state_dict(model):
    new_modules_state_dict = {}
    for name, param in model.named_parameters():
        normalized_name = normalize_new_module_key(name)
        if normalized_name is not None:
            new_modules_state_dict[normalized_name] = param.detach().cpu().clone()
    return new_modules_state_dict


def find_latest_checkpoint(output_dir):
    checkpoint_dirs = []
    if not os.path.isdir(output_dir):
        return None
    final_checkpoint = os.path.join(output_dir, 'checkpoint-final')
    if os.path.isdir(final_checkpoint):
        return final_checkpoint
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path) or not name.startswith('checkpoint-'):
            continue
        suffix = name[len('checkpoint-'):]
        if suffix.isdigit():
            checkpoint_dirs.append((int(suffix), path))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda item: item[0])
    return checkpoint_dirs[-1][1]


def save_new_modules(model, output_dir):
    new_modules_state_dict = collect_new_modules_state_dict(model)
    required_keys = {
        'logit_scale',
        'logit_bias',
        'vision2text.in_proj_weight',
        'vision2text.in_proj_bias',
        'vision2text.out_proj.weight',
        'vision2text.out_proj.bias',
    }
    missing_keys = sorted(required_keys - set(new_modules_state_dict))
    if missing_keys:
        available_keys = sorted(new_modules_state_dict)
        raise RuntimeError(
            f"新模块参数保存失败，缺少 key: {missing_keys}. "
            f"当前已收集 key: {available_keys}"
        )

    save_path = os.path.join(output_dir, "new_modules.pt")
    torch.save(new_modules_state_dict, save_path)
    logger.info(f"New module parameters saved to {save_path}: {sorted(new_modules_state_dict)}")
    return save_path


def load_new_modules_into_model(model, checkpoint_path):
    """Load saved auxiliary module weights into the already initialized model.

    This avoids mutating the shared base model safetensors file. Mutating that file from
    every DeepSpeed rank can race with model loading and lead to rank divergence/hangs.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"new_modules.pt 不存在: {checkpoint_path}")

    pt_state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    loaded = []
    for key, value in pt_state.items():
        normalized_key = normalize_new_module_key(key)
        if normalized_key is None:
            continue
        target_key = normalized_key if normalized_key in model_state else key
        if target_key in model_state:
            model_state[target_key] = value.to(dtype=model_state[target_key].dtype)
            loaded.append(target_key)

    model.load_state_dict(model_state, strict=False)
    if not loaded:
        raise RuntimeError(f"没有从 {checkpoint_path} 加载到任何新模块权重")
    logger.info(f"Loaded new_modules from {checkpoint_path}: {sorted(loaded)}")


class DetailLossTrainer(Seq2SeqTrainer):
    """记录模型 forward 暴露的各子 loss 到 Trainer 日志。"""

    CL_BRANCH_KEYWORDS = ('vision2text', 'logit_scale', 'logit_bias', 'vision_proj', 'text_proj')

    def __init__(self, *args, cl_branch_learning_rate=None, cl_branch_weight_decay=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_branch_learning_rate = None if cl_branch_learning_rate is None else float(cl_branch_learning_rate)
        self.cl_branch_weight_decay = float(cl_branch_weight_decay)

    @staticmethod
    def _find_inner_model(model):
        """穿透 Swift/PEFT/DeepSpeed 包装，找到实际执行 forward 的模型。"""
        stack = [model]
        visited = set()
        while stack:
            candidate = stack.pop()
            candidate_id = id(candidate)
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            if hasattr(candidate, '_log_loss_gen'):
                return candidate
            for attr in ('base_model', 'model', 'module'):
                child = getattr(candidate, attr, None)
                if child is not None:
                    stack.append(child)
        return model

    @staticmethod
    def _get_loss_tensor(loss_output):
        return loss_output[0] if isinstance(loss_output, tuple) else loss_output

    @classmethod
    def _is_cl_branch_param(cls, name):
        return any(keyword in name for keyword in cls.CL_BRANCH_KEYWORDS)

    @staticmethod
    def _should_force_no_decay(name):
        return 'logit_scale' in name or 'logit_bias' in name

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        cl_branch_learning_rate = self.cl_branch_learning_rate or self.args.learning_rate
        cl_branch_weight_decay = self.cl_branch_weight_decay

        optimizer_grouped_parameters = []
        group_specs = [
            (False, True, self.args.learning_rate, self.args.weight_decay),
            (False, False, self.args.learning_rate, 0.0),
            (True, True, cl_branch_learning_rate, cl_branch_weight_decay),
            (True, False, cl_branch_learning_rate, 0.0),
        ]
        named_parameters = list(opt_model.named_parameters())
        for is_cl_branch, use_decay, learning_rate, weight_decay in group_specs:
            params = []
            for name, param in named_parameters:
                if not param.requires_grad:
                    continue
                if self._is_cl_branch_param(name) != is_cl_branch:
                    continue
                param_uses_decay = name in decay_parameters and not self._should_force_no_decay(name)
                if param_uses_decay != use_decay:
                    continue
                params.append(param)
            if params:
                optimizer_grouped_parameters.append({
                    'params': params,
                    'lr': learning_rate,
                    'weight_decay': weight_decay,
                })

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

        if 'params' in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop('params')
        if 'model' in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop('model')
        if 'optimizer_dict' in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop('optimizer_dict')

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        logger.info(
            f"Optimizer groups: base_lr={self.args.learning_rate}, base_weight_decay={self.args.weight_decay}, "
            f"cl_branch_lr={cl_branch_learning_rate}, cl_branch_weight_decay={cl_branch_weight_decay}, "
            f"groups={len(optimizer_grouped_parameters)}"
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_output = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch
        )
        loss_tensor = self._get_loss_tensor(loss_output)

        actual_model = self.accelerator.unwrap_model(model)
        inner_model = self._find_inner_model(actual_model)
        for attr, metric_key in [
            ('_log_loss_gen', 'loss_gen'),
            ('_log_loss_cl', 'loss_cl'),
            ('_log_loss_box', 'loss_box'),
            ('_log_cl_pos_logit', 'cl_pos_logit'),
            ('_log_cl_neg_logit', 'cl_neg_logit'),
            ('_log_logit_scale_exp', 'logit_scale_exp'),
            ('_log_cl_acc', 'cl_acc'),
        ]:
            value = getattr(inner_model, attr, None)
            if value is None:
                continue
            if metric_key not in self._custom_metrics:
                self._custom_metrics[metric_key] = MeanMetric(nan_value=None)
            if isinstance(value, torch.Tensor):
                value = value.detach().to(loss_tensor.device)
            self._custom_metrics[metric_key].update(value)

        return loss_output

def main(config_path: str):
    """
    主训练函数，从 YAML 配置文件中加载所有参数。
    支持 stage1 / stage2-pre / stage2-post 三个阶段。
    """
    # --- 1. 从 YAML 文件加载配置 ---
    config_path = resolve_repo_path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_cfg = config['run_params']
    path_cfg = config['path_params']
    data_cfg = config['data_params']
    path_cfg['base_dir'] = resolve_repo_path(path_cfg['base_dir'])
    deepspeed_path = resolve_repo_path(config['deepspeed_config'])

    # --- 2. 初始化设置 ---
    local_rank = configure_distributed_cuda_device()
    seed_everything(run_cfg['seed'])
    stage = run_cfg['stage']

    # 读取阶段专属配置
    stage_cfg = config['stages'][stage]
    stage_cfg['model_id_or_path'] = resolve_repo_path(stage_cfg['model_id_or_path'])
    stage_cfg['dataset'] = resolve_repo_paths(stage_cfg['dataset'])
    if 'new_modules_pt' in stage_cfg:
        stage_cfg['new_modules_pt'] = resolve_repo_path(stage_cfg['new_modules_pt'])
    if 'lora_checkpoint' in stage_cfg:
        stage_cfg['lora_checkpoint'] = resolve_repo_path(stage_cfg['lora_checkpoint'])
    if 'lora_checkpoints' in stage_cfg:
        stage_cfg['lora_checkpoints'] = resolve_repo_paths(stage_cfg['lora_checkpoints'])
    model_id_or_path = stage_cfg['model_id_or_path']
    system = stage_cfg.get('system_prompt', '')
    dataset_paths = stage_cfg['dataset']

    logger.info(f"Stage: {stage}, System Prompt: '{system}'")

    # --- 3. 设置输出目录 ---
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = os.path.join(path_cfg['base_dir'], stage, current_time)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Output directory: {output_dir}')

    # --- 4. 初始化模型和处理器 ---
    # The custom Qwen2.5-VL class creates the auxiliary CL modules in __init__.
    # Do not edit the shared base safetensors in distributed runs; every rank should
    # load the same immutable base model, then load previous-stage auxiliary weights
    # directly into memory when needed.
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
    model.stage2_post_cl_weight = float(stage_cfg.get('stage2_post_cl_weight', 1.0))
    logger.info(f"Stage2-post CL weight: {model.stage2_post_cl_weight}")

    if stage in {'stage2-pre', 'stage2-post'}:
        load_new_modules_into_model(model, stage_cfg.get('new_modules_pt', ''))

    if stage in {'stage2-pre', 'stage2-post'}:
        model = merge_lora_checkpoints(model, get_lora_checkpoints(stage_cfg), stage)
        model.stage2_post_cl_weight = float(stage_cfg.get('stage2_post_cl_weight', 1.0))

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

    # --- 5. 配置 LoRA ---
    target_modules_regex = get_multimodal_target_regex(
        model,
        freeze_llm=stage_cfg['freeze_llm'],
        freeze_vit=stage_cfg['freeze_vit'],
        freeze_aligner=stage_cfg['freeze_aligner']
    )

    # PEFT 只有在 target_modules 是字符串时才按正则匹配。
    # 这里让 LLM/aligner 的线性层走 LoRA；新增的 vision2text/logit_scale/logit_bias 在下方手动解冻做全参数训练。
    target_modules = target_modules_regex

    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=stage_cfg['lora_rank'],
        lora_alpha=stage_cfg['lora_alpha'],
        target_modules=target_modules
    )
    # --- 6. 准备当前阶段 LoRA ---
    model = Swift.prepare_model(model, lora_config)

    # 解冻标量参数和 MultiheadAttention（LoRA 不适用于这些模块，需要直接训练）
    for name, param in model.named_parameters():
        if any(k in name for k in ['logit_scale', 'logit_bias', 'vision2text']):
            param.requires_grad = True

    logger.info(f'LoRA Config: {lora_config}')
    logger.info(f'Trainable Parameters: {get_model_parameter_info(model)}')

    # --- 7. 加载和预处理数据集 ---
    train_dataset, _ = load_dataset(
        dataset_paths,
        split_dataset_ratio=data_cfg['split_dataset_ratio'],
        num_proc=data_cfg['num_proc'],
        seed=run_cfg['seed']
    )
    train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=run_cfg['seed'])
    logger.info(f'Train dataset loaded: {train_dataset}')

    # --- 8. 设置训练参数 ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        data_seed=run_cfg['seed'],
        deepspeed=deepspeed_path,
        learning_rate=stage_cfg['learning_rate'],
        num_train_epochs=stage_cfg['num_train_epochs'],
        max_steps=stage_cfg.get('max_steps', -1),
        save_steps=stage_cfg['save_steps'],
        save_total_limit=stage_cfg['save_total_limit'],
        gradient_checkpointing=stage_cfg['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False} if stage_cfg['gradient_checkpointing'] else None,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=1,
        weight_decay=stage_cfg.get('weight_decay', 0.01),
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        eval_strategy='steps',
        eval_steps=100000,
        gradient_accumulation_steps=1,
        metric_for_best_model='loss',
        logging_steps=5,
        # Keep worker count configurable. stage1/stage2-pre now avoid CUDA model
        # execution in the template pre-hook; stage2-post still uses the template path.
        dataloader_num_workers=stage_cfg.get('dataloader_num_workers', 0),
        remove_unused_columns=False,
        bf16=True,
        tf32=True,
    )

    # --- 9. 初始化并开始训练 ---
    trainer = DetailLossTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        template=template,
        cl_branch_learning_rate=stage_cfg.get('cl_branch_learning_rate'),
        cl_branch_weight_decay=stage_cfg.get('cl_branch_weight_decay', 0.0),
    )

    trainer.train()

    final_checkpoint_dir = None
    if stage_cfg.get('save_final_checkpoint', False):
        final_checkpoint_dir = os.path.join(output_dir, 'checkpoint-final')
        trainer.save_model(final_checkpoint_dir)
        logger.info(f"Final checkpoint saved to {final_checkpoint_dir}")

    # --- 10. 训练后保存新模块参数 ---
    if trainer.is_world_process_zero():
        wrapped_model = getattr(trainer, 'model_wrapped', None) or trainer.model
        save_model = trainer.accelerator.unwrap_model(wrapped_model)
        new_modules_path = save_new_modules(save_model, output_dir)

        latest_checkpoint = final_checkpoint_dir or find_latest_checkpoint(output_dir)
        artifacts = {
            'stage': stage,
            'output_dir': output_dir,
            'new_modules_pt': new_modules_path,
            'latest_checkpoint': latest_checkpoint,
        }
        artifacts_path = os.path.join(output_dir, 'artifacts.yaml')
        with open(artifacts_path, 'w') as f:
            yaml.safe_dump(artifacts, f, allow_unicode=True, sort_keys=False)
        logger.info(f"Stage artifacts saved to {artifacts_path}: {artifacts}")

    # --- 12. 将配置文件保存到输出目录以备查阅 ---
    if trainer.is_world_process_zero():
        final_config_path = os.path.join(output_dir, 'config.yaml')
        shutil.copy(config_path, final_config_path)
        logger.info(f"Configuration file saved to {final_config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default='./code/train_stage1or2.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args, _ = parser.parse_known_args()
    if args.local_rank >= 0:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    main(args.config_path)
