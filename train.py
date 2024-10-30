import torch
from tqdm import tqdm
import numpy as np
import copy
from datetime import datetime
import os
import yaml
import random

# Function to set the seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

set_seed(42)

from torch.utils.data import DataLoader
from dataset import LoadDataAndLabel
from sampler import RandomSampler
from model.torch_hub_models import Hub_Model
from model.CNN import Custom_CNN
from sklearn.metrics import accuracy_score


# Load and parse configuration from YAML
def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config()

# Extract settings
audio_config = config['audio']
model_config = config['model']
training_config = config['training']
datasets_config = config['datasets']
augmentation_resources = config['augmentation_resources']

sampling_rate = audio_config['sampling_rate']
n_fft, window_len, step_size  = audio_config['window_configs'][sampling_rate]

task = config['task']['type']
if task == "gender":
    n_classes = 2
elif task == "dialect":
    n_classes = 3
else:
    raise ValueError(f"Unsupported task type: {task}. Valid options are 'gender' or 'dialect'.")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model_type = model_config['type']
if model_type == 'CNN':
    # Use the custom CNN-based model
    model = Custom_CNN(
        n_classes=n_classes,
        n_mels=audio_config['n_mels'],
        dropout_rate=model_config['dropout'],
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window_len,
        step=step_size
    ).to(device)
else:
    # Use the Model with a pretrained backbone
    try:
        model = Hub_Model(
            n_classes=n_classes,
            n_mels=audio_config['n_mels'],
            dropout_rate=model_config['dropout'],
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            window=window_len,
            step=step_size,
            model_type=model_type,
            version=model_config['versions'][model_type]
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_type}' from PyTorch Hub: {e}")


# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
criterion = torch.nn.functional.nll_loss

# Load datasets
train_dataset = LoadDataAndLabel(
    dataset=datasets_config['trainset'],
    audio_length=audio_config['length'],
    sampling_rate=sampling_rate,
    window_len=window_len,
    step_size=step_size,
    task_type=task,
    enable_augmentation=training_config['enable_augmentation'],
    rir_path=augmentation_resources['rir_path'],
    noise_path=augmentation_resources['noise_path']
)

val_dataset = LoadDataAndLabel(
    dataset=datasets_config['valset'],
    audio_length=audio_config['length'],
    sampling_rate=sampling_rate,
    window_len=window_len,
    step_size=step_size,
    task_type=task,
    enable_augmentation=False
)

loader_val = DataLoader(val_dataset, batch_size=training_config['batch_size'], num_workers=2, shuffle=False)


def evaluate(model, dataloader):
    """Evaluate the model on the validation set."""
    model.eval()
    true_labels, predicted_labels, losses = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs.to(device))
            labels_device = torch.LongTensor(labels).to(device)
            
            loss = criterion(outputs, labels_device)
            losses.append(loss.item())

            true_labels.extend(torch.LongTensor(labels).to('cpu').numpy().tolist())
            predicted_labels.extend(np.argmax(outputs.to('cpu').numpy(), axis=1).tolist())

    return np.mean(losses), accuracy_score(true_labels, predicted_labels)

def train(model, dataloader):
    """Train the model for one epoch."""
    model.train()
    true_labels, predicted_labels, losses = [], [], []
    
    for inputs, labels in tqdm(dataloader):
        outputs = model(inputs.to(device))
        labels_device = torch.LongTensor(labels).to(device)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels_device)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        true_labels.extend(labels_device.tolist())
        predicted_labels.extend(torch.argmax(outputs, dim=1).tolist())

    return np.mean(losses), accuracy_score(true_labels, predicted_labels)

def save_checkpoint(state, filename):
    """Save the model checkpoint."""
    torch.save(state, filename)

def run_training():
    """Run the full training loop."""
    # Create a timestamped folder for checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/{task}/{model_type}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training {model_type} classification")

    best_checkpoints = []

    for epoch in range(training_config['num_epochs']):
        # Create a DataLoader for training with batch sampling
        loader_train = DataLoader(
            train_dataset, 
            batch_sampler=RandomSampler(train_dataset, task, n_classes, batch_size=training_config['batch_size']), 
            num_workers=2, 
            pin_memory=True
        )

        train_loss, train_acc = train(model, loader_train)
        val_loss, val_acc = evaluate(model, loader_val)

        print(f"EPOCH {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save the last checkpoint every epoch
        save_checkpoint(model.state_dict(), os.path.join(checkpoint_dir, "last.pth"))

        # Save the top 3 best checkpoints based on validation accuracy
        if len(best_checkpoints) < 3 or val_acc > best_checkpoints[-1][0]:
            best_checkpoints.append((val_acc, copy.deepcopy(model.state_dict())))
            best_checkpoints = sorted(best_checkpoints, key=lambda x: x[0], reverse=True)[:3]

            # Save the best checkpoints
            for idx, (acc, state) in enumerate(best_checkpoints, 1):
                checkpoint_path = os.path.join(checkpoint_dir, f"best_{idx}.pth")
                save_checkpoint(state, checkpoint_path)

if __name__ == "__main__":
    run_training()