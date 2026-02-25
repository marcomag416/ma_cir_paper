import os
import torch

from evals.cirr_eval import cirr_test_alpha
from evals.circo_eval import circo_test_alpha
from evals.fashioniq_eval import fashioniq_test_alpha
from evals.ma_cir_eval import macir_test_alpha
from models import TwoEncoderVLM, AutoConfig, AutoModel
import json

from matplotlib import pyplot as plt

def test_alpha(
    model: TwoEncoderVLM,
    alphas: list[int] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    skip_datasets: list[str] = [],
    batch_size: int = 64,
    num_workers: int = 4,
    use_tqdm: bool = False,
):
    if "fashioniq" not in skip_datasets:
        fashioniq_metrics = fashioniq_test_alpha(
            model=model,
            alphas=alphas,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=use_tqdm,
        )
    else:
        fashioniq_metrics = None
        
    if "circo" not in skip_datasets:
        circo_metrics = circo_test_alpha(
            model=model,
            alphas=alphas,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=use_tqdm,
        )
    else:
        circo_metrics = None

    if "cirr" not in skip_datasets:
        cirr_metrics = cirr_test_alpha(
            model=model,
            alphas=alphas,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=use_tqdm,
        )
    else:
        cirr_metrics = None

    if "macir" not in skip_datasets:
        macir_metrics = macir_test_alpha(
            model=model,
            alphas=alphas,
            batch_size=batch_size,
            num_workers=num_workers,
            use_tqdm=use_tqdm,
        )
    else:
        macir_metrics = None

    return circo_metrics, cirr_metrics, macir_metrics, fashioniq_metrics

def plot_alpha_sweep(alphas, metric, metric_name, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(alphas, metric, marker="o")
    plt.xlabel("Alpha")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Alpha")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    checkpoint_path = args.checkpoint_path
    model_config_path = os.path.relpath(os.path.join(checkpoint_path, os.pardir))

    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config path does not exist: {model_config_path}")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")

    dirname_ck_component = os.path.basename(checkpoint_path.rstrip('/')) if not args.zero_shot else "zero-shot"
    dirname_config_component = os.path.basename(model_config_path.rstrip('/'))
    newdirname = dirname_config_component + "-" + dirname_ck_component 
    output_path = os.path.join(args.output_path, newdirname)
    os.makedirs(output_path, exist_ok=True)

    #load model
    print("Loading model config from:", model_config_path)    
    config = AutoConfig.from_pretrained(model_config_path)
    model = AutoModel.from_config(config)

    if args.zero_shot:
        print("Zero-shot model specified; skipping loading of adapter or full model weights.")

    elif not args.no_peft:
        adapter_path = os.path.join(checkpoint_path, "lora_adapter")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"PEFT adapter path does not exist: {adapter_path}")
        print("Loading PEFT adapter from:", adapter_path)
        model = model.apply_peft_from_pretrained(adapter_path)
        
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint path does not exist: {checkpoint_path}")
        print("Loading full model weights from:", checkpoint_path)
        model.from_pretrained(checkpoint_path)

    #compute and save metrics to file
    model.to(device)
    circo_metrics, cirr_metrics, macir_metrics, fashioniq_metrics = test_alpha(
        model=model,
        alphas=args.alphas,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tqdm=args.tqdm,
        skip_datasets=args.skip_datasets,
    )

    if fashioniq_metrics is not None:
        with open(os.path.join(output_path, "alpha_sweep_fashioniq.json"), "w") as f:
            json.dump(fashioniq_metrics, f, indent=4)
        alphas = []
        fashioniq_avg = []
        for alpha, m_dict in fashioniq_metrics.items():
            alphas.append(float(alpha))
            fashioniq_avg.append(m_dict["avg_recall_at@10"])
        plot_alpha_sweep(alphas, fashioniq_avg, "FashionIQ Avg R@10", os.path.join(output_path, "alpha_sweep_fashioniq_average.png"))

    if circo_metrics is not None:
        with open(os.path.join(output_path, "alpha_sweep_circo.json"), "w") as f:
            json.dump(circo_metrics, f, indent=4)
        alphas = []
        circo_map_at5 = []
        for alpha, m_dict in circo_metrics.items():
            alphas.append(float(alpha))
            circo_map_at5.append(m_dict["mAP_at5"])
        plot_alpha_sweep(alphas, circo_map_at5, "CIRCO mAP@5", os.path.join(output_path, "alpha_sweep_circo_map_at5.png"))

    if cirr_metrics is not None:
        with open(os.path.join(output_path, "alpha_sweep_cirr.json"), "w") as f:
            json.dump(cirr_metrics, f, indent=4)
        alphas = []
        cirr_rec_at1 = []
        for alpha, m_dict in cirr_metrics.items():
            alphas.append(float(alpha))
            cirr_rec_at1.append(m_dict["recall_at1"])
        plot_alpha_sweep(alphas, cirr_rec_at1, "CIRR R@1", os.path.join(output_path, "alpha_sweep_cirr_rec_at1.png"))


    if macir_metrics is not None:
        with open(os.path.join(output_path, "alpha_sweep_macir.json"), "w") as f:
            json.dump(macir_metrics, f, indent=4)
        alphas = []
        macir_avg = []
        for alpha, m_dict in macir_metrics.items():
            alphas.append(float(alpha))
            macir_avg.append(m_dict["avg"]["recall_at1"])
        plot_alpha_sweep(alphas, macir_avg, "MaCIR Avg", os.path.join(output_path, "alpha_sweep_macir_average.png"))

    #todo: create plots of main score vs alpha for each dataset




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model using slerp and sweeping through alpha values.")
    # Add arguments as needed
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint. Model config will be loaded from the parent directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm for progress bars.")
    parser.add_argument("--output_path", type=str, default="results/", help="Path to save the evaluation metrics.")
    parser.add_argument("--no_peft", action='store_true', help="Do not use PEFT model.")
    parser.add_argument("--skip_datasets", nargs='*', default=[], help="List of datasets to skip during evaluation.")
    parser.add_argument("--zero-shot", action='store_true', help="Indicates if the model is zero-shot. Adapter or full model weights are ignored if set. Model config is still loaded from checkpoint path parent directory.")
    parser.add_argument("--alphas", nargs='*', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], help="List of alpha values to test for slerp fusion.")
    args = parser.parse_args()
    main(args)

    
    


