import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
import pandas as pd
import matplotlib.pyplot as plt

# Load your model and tokenizer
def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load datasets
def load_datasets():
    datasets = {
        "TruthfulQA": load_dataset("truthful_qa"),
        "HellaSwag": load_dataset("hellaswag"),
        "MMLU": load_dataset("mmlu"),
        "StereoSet": load_dataset("stereoset"),
        "CrowS-Pairs": load_dataset("crows_pairs"),
        "RealToxicityPrompts": load_dataset("real_toxicity_prompts")
    }
    return datasets

# Load metrics
def load_metrics():
    metrics = {
        "TruthfulQA": load_metric("accuracy"),
        "HellaSwag": load_metric("accuracy"),
        "MMLU": load_metric("accuracy"),
        "StereoSet": load_metric("accuracy"),
        "CrowS-Pairs": load_metric("accuracy"),
        "RealToxicityPrompts": load_metric("accuracy")
    }
    return metrics

def evaluate_model(model, tokenizer, datasets, metrics, dataset_name=None):
    results = {}
    
    if dataset_name:
        datasets = {dataset_name: datasets[dataset_name]}
    
    for dataset_name, dataset in datasets.items():
        print(f"Evaluating on {dataset_name}...")

        all_predictions = []
        all_labels = []

        for sample in dataset['test']:
            prompt = sample['prompt']
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            label = sample['label']

            all_predictions.append(prediction)
            all_labels.append(label)

        metric = metrics[dataset_name]
        score = metric.compute(predictions=all_predictions, references=all_labels)
        results[dataset_name] = score

    return results

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "your-preferred-llm"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else None

    model, tokenizer = load_model_and_tokenizer(model_name)
    datasets = load_datasets()
    metrics = load_metrics()

    results = evaluate_model(model, tokenizer, datasets, metrics, dataset_name)

    # Display results
    df_results = pd.DataFrame.from_dict(results, orient='index')
    print(df_results)

    # Visualize results
    df_results.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Evaluation Results')
    plt.ylabel('Scores')
    plt.show()
