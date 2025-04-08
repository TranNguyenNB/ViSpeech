import torch
from model.torch_hub_models import Hub_Model
from model.CNN import Custom_CNN
from torch.utils.data import DataLoader
from dataset import LoadData
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

import yaml

def load_config(file_path='config_infer.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config()


# Extracting the values from the config
model_configs = config['model']
task = model_configs['task']
model_type = model_configs['type']
length = model_configs['length']
sr = model_configs['sampling_rate']
n_mels = model_configs['n_mels']
n_fft, window_len, step_size = model_configs['window_configs'][sr]
checkpoint = model_configs.get('checkpoint')


datasets = config['datasets']['paths']

if task == "gender":
    n_classes = 2
elif task == "dialect":
    n_classes = 3
else:
    raise ValueError(f"Unsupported task type: {task}. Valid options are 'gender' or 'dialect'.")

# Determine the device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the appropriate model based on model_type
if model_type == 'CNN':
    # Use the custom CNN-based model
    model = Custom_CNN(
        n_classes=n_classes,
        n_mels=n_mels,
        sampling_rate=sr,
        n_fft=n_fft,
        window=window_len,
        step=step_size
    ).to(device)
else:
    # Use the Model with a pretrained backbone
    try:
        model = Hub_Model(
            n_classes=n_classes,
            n_mels=n_mels,
            sampling_rate=sr,
            n_fft=n_fft,
            window=window_len,
            step=step_size,
            model_type=model_type,
            version=model_configs['versions'][model_type]
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_type}' from PyTorch Hub: {e}")

# Load the model checkpoint    
model.load_state_dict(torch.load(checkpoint))
model.eval()

def infer(model, dataloader):
    """ Inference """
    predicted_classes, probabilities, audio_files = [], [], []

    with torch.no_grad():
        for audio, filename in tqdm(dataloader):
            proba = torch.softmax(model(audio.to(device)), dim=1).to('cpu').numpy()
            predicted_class = np.argmax(model(audio.to(device)).to('cpu').numpy(), axis=1)
            
            predicted_classes.extend(predicted_class.tolist())
            probabilities.extend(proba.tolist())
            audio_files.extend(filename)

    df = pd.DataFrame({
        'audio_file': audio_files,
        'probability': probabilities,
        'class': predicted_classes
    })

    cols = [f'proba_class_{i}' for i in range(len(probabilities[0]))]
    df[cols] = pd.DataFrame(df['probability'].tolist(), index=df.index)
    df.drop(columns='probability', inplace=True)
    return df

# Iterate through each dataset path and perform inference
for dataset in datasets:
    print(f"------Predicting {dataset}")
    # Load data
    set = LoadData(
        dataset=dataset,
        audio_length=length,
        sampling_rate=sr,
        window_len=window_len,
        step_size=step_size,
    )
    dataloader = DataLoader(set, batch_size=1, num_workers=2, shuffle=False)

    # Infer
    df = infer(model, dataloader)

    # Save the results
    if not os.path.exists("inference_result"):
        os.makedirs("inference_result")
    result_path = f"inference_result/{dataset.split('/')[-1]}.csv"
    df.to_csv(result_path, index=False)
    print(f"Done. Saved to {result_path}\n")


