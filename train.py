import os
import time
from types import SimpleNamespace
from peft import LoraConfig, LoraModel
import torch
import wandb
from transformers import Trainer, TrainingArguments
from evals.circo_eval import evaluate_circo
from evals.cirr_eval import evaluate_cirr
from evals.ma_cir_eval import evaluate_macir
from evals.simat_eval import evaluate_simat
from losses import build_loss_fn
from models import build_clip, TwoEncoderVLM
import argparse
import os
import json
from datasets.mscoco import build_mscoco_dataset
from evals.metrics import compute_modality_gap_metrics, compute_statistic_metrics
from torch.optim import AdamW
from utils.dict import prepend_key_to_dict

class CustomLossTrainer(Trainer):
    """Hugging Face Trainer that uses a user-provided loss function.

    The provided `loss_fn` is expected to have signature:
        loss_fn(outputs, inputs) -> torch.Tensor

    Where `outputs = model(**inputs)` and `inputs` is the batch dict from the data collator.
    """

    def __init__(self, *args, loss_fn=None, custom_eval_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_fn is None:
            raise ValueError("loss_fn must be provided for CustomLossTrainer")
        self.loss_fn = loss_fn
        self.custom_eval_func = custom_eval_func # Store the custom function
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = self.loss_fn(outputs, inputs, num_items_in_batch=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step that rigidly ensures strict Tensor outputs for Accelerate.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
                loss = self.loss_fn(outputs, inputs)
        
        # Safely extract embeddings
        vision_embeds = outputs.get('vision_embeds')
        text_embeds = outputs.get('text_embeds')
        
        valid_device = loss.device
        
        # SANITIZATION: Ensure everything is a Tensor
        if vision_embeds is None:
            vision_embeds = torch.empty(0, device=valid_device)
        elif not isinstance(vision_embeds, torch.Tensor):
            # If it's a float/list, value-wrap it
            vision_embeds = torch.tensor(vision_embeds, device=valid_device)
            
        if text_embeds is None:
            text_embeds = torch.empty(0, device=valid_device)
        elif not isinstance(text_embeds, torch.Tensor):
            text_embeds = torch.tensor(text_embeds, device=valid_device)

        # Create explicit dummy labels
        # (Trainer will try to pad these, so they must be tensors)
        batch_len = vision_embeds.shape[0] if vision_embeds.ndim > 0 else 1
        dummy_labels = torch.zeros(batch_len, device=valid_device)
        
        # Return strict (Tensor, Tuple[Tensor, Tensor], Tensor)
        return (loss, (vision_embeds, text_embeds), dummy_labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        """
        Overridden evaluate method to run standard validation AND custom task evaluation.
        """

        custom_metrics = {}
        if self.custom_eval_func:
            custom_metrics = self.custom_eval_func(self)
            
            wandb.log(custom_metrics, step=self.state.global_step, commit=False)
            print(custom_metrics)
			


        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **kwargs)

        if custom_metrics:
            metrics.update(custom_metrics)

        return metrics


def compute_metrics_fn(eval_pred):
	predictions, labels = eval_pred
	
	# Unpack safely
	if isinstance(predictions, tuple):
		image_features, text_features = predictions
	else:
		# Fallback if Trainer passed something else
		print(f"[WARN] Predictions format unexpected: {type(predictions)}")
		return {}
		
	# Convert numpy to tensor if needed
	if not isinstance(image_features, torch.Tensor):
		image_features = torch.tensor(image_features)
	if not isinstance(text_features, torch.Tensor):
		text_features = torch.tensor(text_features)

	mg_metrics = compute_modality_gap_metrics(image_features, text_features)
	stat_metrics = compute_statistic_metrics(image_features, text_features)
	metrics = {}
	metrics.update(mg_metrics)
	metrics.update(stat_metrics)
	return metrics

def evaluate_on_tasks(trainer: Trainer):
	"""
	Evaluate the model on custom tasks and return metrics.

	Args:
		trainer (Trainer): The Hugging Face Trainer with the model to evaluate.
	Returns:
		dict: Dictionary of evaluation metrics.
	"""
	metrics = {}
	model = trainer.model

	simat_metrics = evaluate_simat(
		model=model,
		batch_size=trainer.args.per_device_eval_batch_size,
		num_workers=trainer.args.dataloader_num_workers,
		tqdm=not trainer.args.disable_tqdm,
		accelerator=trainer.accelerator
	)
	metrics.update(prepend_key_to_dict("simat/", simat_metrics))

	if not trainer.is_in_train:
		# Only compute these if not in training mode (to save time)
		circo_metrics = evaluate_circo(
			model=model,
			fusion_type="sum",
			batch_size=trainer.args.per_device_eval_batch_size,
			num_workers=trainer.args.dataloader_num_workers,
			tqdm=not trainer.args.disable_tqdm,
			accelerator=trainer.accelerator
		)
		metrics.update(prepend_key_to_dict("circo/", circo_metrics))

	cirr_metrics = evaluate_cirr(
		model=model,
		fusion_type="sum",
		batch_size=trainer.args.per_device_eval_batch_size,
		num_workers=trainer.args.dataloader_num_workers,
		tqdm=not trainer.args.disable_tqdm,
		accelerator=trainer.accelerator
	)
	metrics.update(prepend_key_to_dict("cirr/", cirr_metrics))

	ma_cir_metrics = evaluate_macir(
		model=model,
		eval_level="full",
		split="",
		batch_size=trainer.args.per_device_eval_batch_size,
		num_workers=trainer.args.dataloader_num_workers,
		tqdm=not trainer.args.disable_tqdm,
		fusion_type="sum",
		accelerator=trainer.accelerator
	)
	metrics.update(prepend_key_to_dict("macir/", ma_cir_metrics))
	
	return metrics

def train(args):
	"""
	Post-train a Hugging Face model using a custom loss and log to Weights & Biases.

	Expected keys in `args` (dict-like or argparse.Namespace):
	  - model: Preloaded HF model (e.g., BlipForImageTextRetrieval)
	  - loss_fn: Callable taking (outputs, inputs, num_items_in_batch) and returning a scalar loss
	  - train_dataset: HF-style dataset providing batches via data collator
	  - eval_dataset (optional): validation dataset
	  - data_collator (optional): callable collator to build batch dicts
	  - output_dir (str): directory to save checkpoints
	  - batch_size (int)
	  - num_epochs (int)
	  - lr (float)
	  - weight_decay (float)
	  - warmup_ratio (float)
	  - logging_steps (int)
	  - save_strategy (str)
	  - save_steps (int)
	  - save_total_limit (int)
	  - logging_steps (int)
	  - seed (int)
	  - fp16 (bool)
	  - gradient_accumulation_steps (int)
	  - max_steps (int, optional): if >0 overrides epochs
	  - wandb_project (str, optional)
	  - wandb_run_name (str, optional)
	  - report_to (str, optional): defaults to 'wandb'
	  - num_workers (int, optional): number of data loading workers
	  - tqdm (bool, optional): enable tqdm progress bars

	Returns: `Trainer` after training completes.
	"""

	# Helper to access args regardless of dict or Namespace
	def get(key, default=None):
		return getattr(args, key, args.get(key, default) if isinstance(args, dict) else default)

	model = get("model")
	loss_fn = get("loss_fn")
	
	train_dataset = get("train_dataset")
	eval_dataset = get("eval_dataset", None)
	data_collator = get("data_collator", None)
	output_dir = get("output_dir", "outputs/run")

	os.makedirs(output_dir, exist_ok=True)	

	if model is None or loss_fn is None or train_dataset is None:
		raise ValueError("args must include 'model', 'loss_fn', and 'train_dataset'")


	# Initialize Weights & Biases before Trainer to capture configs
	wandb_project = get("wandb_project", "ma_loss_paper")
	wandb_run_name = get("name", "run")
	wandb.init(project=wandb_project, name=wandb_run_name)

	training_args = TrainingArguments(
		output_dir=output_dir,
		per_device_train_batch_size=get("batch_size", 128),
		per_device_eval_batch_size=get("batch_size", 128),
		gradient_accumulation_steps=get("gradient_accumulation_steps", 1),
		num_train_epochs=get("num_epochs", 3) if get("max_steps", -1) <= 0 else 1,
		max_steps=get("max_steps", -1),
		learning_rate=get("lr", 1e-6),
		weight_decay=get("weight_decay", 0.1),
		warmup_ratio=get("warmup_ratio", 0.0),

		eval_strategy = get("save_strategy", "epoch"),
		eval_steps = get("save_steps", 500),
		logging_steps=get("logging_steps", 50),
		save_steps=get("save_steps", 500),
		save_strategy=get("save_strategy", "epoch"),
		save_total_limit=get("save_total_limit", 10),
		seed=get("seed", 42),
		fp16=get("fp16", False),
		report_to="wandb",
		run_name=wandb_run_name,
		remove_unused_columns =False,
		dataloader_num_workers=get("num_workers", 4),
		disable_tqdm=not get("tqdm", False),
		eval_on_start=True,
		
		# CRITICAL: ensure this is False so our prediction_step runs fully
		prediction_loss_only=False, 
	)

	trainer = CustomLossTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=data_collator,
		loss_fn=loss_fn,
		custom_eval_func=evaluate_on_tasks,
		compute_metrics=compute_metrics_fn,
		optimizers=(get("optimizer", None), get("lr_scheduler", None))
	)
	
	trainer.train()

	# Save final artifacts
	trainer.save_model(output_dir)
	if hasattr(model, "processor") and model.processor is not None:
		try:
			model.processor.save_pretrained(output_dir)
		except Exception:
			pass

	wandb.finish()
	return trainer


def main(args):
	if not os.path.isfile(args.config):
		raise ValueError(f"Configuration file '{args.config}' not found.")
    
	with open(args.config, "r") as f:
		config = json.load(f)

	if type(config) is not list:
		raise ValueError(f"Configuration file must be a list of objects. Found {type(config)}")
    
	for run_config in config:
		if run_config.get("name") == args.run:
			break
	else:
		raise ValueError(f"Run configuration named '{args.run}' not found in the configuration file.")
    
	if torch.cuda.is_available():
		print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
	else:
		print("No GPU available, using CPU.")

	print (f"Starting run: {args.run} with config: {run_config}")

    # Merge command-line arguments into run configuration
	run_config["output_dir"] = args.output_dir
	run_config["cache_dir"] = args.cache_dir
	run_config["debug"] = args.debug
	run_config["num_workers"] = args.num_workers
	run_config["tqdm"] = args.tqdm
	run_config["save_strategy"] = args.save_strategy
	run_config["save_steps"] = args.save_steps

	model_name = run_config.get("model_name")
	
	if model_name.startswith("CLIP_"):
		clip_model_name = model_name.split("CLIP_")[1]
		vision_model, image_transform, text_model, tokenizer = build_clip(SimpleNamespace(
			clip_model_name=clip_model_name,
			cache_dir=run_config.get("cache_dir", ".cache"),
			mixed_precision= "fp16" if run_config.get("mixed_precision", False) else None,
		))
		
		if run_config.get("temperature") == "learnable":
			trainable_temp = True
			logit_scale = 100
		else:
			trainable_temp = False
			logit_scale = run_config.get("logit_scale", 100)

		model = TwoEncoderVLM(
			vision_model=vision_model,
			text_model=text_model,
			image_processor=image_transform,
			tokenizer=tokenizer,
			logit_scale=logit_scale,
			trainable_temp=trainable_temp,
			proj_dim=run_config.get("proj_dim", None),
		)

		if run_config.get("use_lora", False):
			config = LoraConfig(
				task_type="FEATURE_EXTRACTION",
				r=run_config.get("lora_r", 8),
				lora_alpha=run_config.get("lora_alpha", 32),
				lora_dropout=run_config.get("lora_dropout", 0.01),
				target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "text_projection", "visual_projection", "position_embedding", "token_embedding", "patch_embedding"],
			)

			model = LoraModel(model, config, "lora_adapter")

	loss_name = run_config.get("loss", "clip")
	loss_fn = build_loss_fn(loss_name, **run_config.get("loss_params", {}))

	dataset_name = run_config.get("dataset_name", "mscoco")
	if dataset_name == "mscoco":
		train_dataset = build_mscoco_dataset(
			split="train",
			image_transform=image_transform,
			caption_transform=tokenizer,
		)
		eval_dataset = build_mscoco_dataset(
			split="val",
			image_transform=image_transform,
			caption_transform=tokenizer,
		)
	else:
		raise ValueError(f"Unsupported dataset: {dataset_name}")
	
	optimizer = AdamW(model.parameters(), lr=run_config.get("lr", 1e-6))

	scheduler_name = run_config.get("scheduler", "none")
	if scheduler_name == "none":
		from transformers import get_constant_schedule
		scheduler = get_constant_schedule(optimizer)
	elif scheduler_name == "cosine":
		from transformers import get_cosine_schedule_with_warmup
		scheduler = get_cosine_schedule_with_warmup(optimizer)
	else:
		raise ValueError(f"Unsupported scheduler: {scheduler_name}")
	
	output_dir = os.path.join(run_config.get("output_dir", "outputs/run"), run_config.get("name", "run"))
	os.makedirs(output_dir, exist_ok=True)

	run_config["model"] = model
	run_config["loss_fn"] = loss_fn
	run_config["train_dataset"] = train_dataset
	run_config["eval_dataset"] = eval_dataset
	run_config["data_collator"] = train_dataset.collate_fn
	run_config["output_dir"] = output_dir
	run_config["lr_scheduler"] = scheduler
	run_config["optimizer"] = optimizer


	train(run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Name of the run configuration to execute.", dest="run")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file.", dest="config")
    parser.add_argument("--output_dir", type=str, default="outputs/run", help="Directory to save outputs.", dest="output_dir")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory to cache models and datasets.", dest="cache_dir")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.", dest="debug")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.", dest="num_workers")
    parser.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bars.", dest="tqdm")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy for checkpoints.", dest="save_strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving checkpoints.", dest="save_steps")
    args = parser.parse_args()
    main(args)