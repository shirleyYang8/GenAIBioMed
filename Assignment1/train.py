import os 
import numpy as np
import torch
import torch.nn as nn 

from dataset import get_dataloader, alphabet
from model import ProGenForCausalLM

torch.manual_seed(0)

# --- Training the Model ---
# Hyperparameters! Don't change
LEARNING_RATE = 1e-3
LR_SCHEDULER_UPDATES = 4000
EPOCHS = 10

# you might modify based on the computing resources you used
BATCH_SIZE = 32

PRETRAINED_MODEL = "progen2-small"
PRETRAINED_MODEL_PATH = "pretrained_model"
DATA_PATH = "data"
OUTPUT_MODEL_PATH = "models"

train_dataloader = get_dataloader(DATA_PATH, 'train', BATCH_SIZE, shuffle=True)
valid_dataloader = get_dataloader(DATA_PATH, 'valid', BATCH_SIZE, shuffle=False)
pad_idx = alphabet.pad()

model = ProGenForCausalLM.from_pretrained(os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL)).to('cuda')
# TODO: Define optimizer, please using Adam
raise NotImplementedError("Need to be implemented for Adam optimizer")

decay_factor = LEARNING_RATE * LR_SCHEDULER_UPDATES ** 0.5

print("\nStarting model training...")
num_updates = 0
best_valid_loss = np.inf 
for epoch in range(EPOCHS):
    
    for inputs, targets in train_dataloader:
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        # TODO: Get model outputs based on inputs - calling model forward
        raise NotImplementedError("Need to be implemented for calling model forward function")
        
        # Calculate loss
        loss = torch.mean(-torch.log(outputs.gather(dim=-1, index=targets.unsqueeze(-1))).squeeze(-1) * (targets!=pad_idx))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # decay lr
        num_updates += 1
        lr = decay_factor * num_updates ** -0.5
        optimizer.lr = lr
        
        
        print(f'Epoch={epoch+1}, updates={num_updates}, train_loss={loss.item()}, lr={lr}')
    
    valid_losses = []
    for inputs, targets in valid_dataloader:
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')

        # Forward pass
        # TODO: Get model outputs based on inputs - calling model forward
        raise NotImplementedError("Need to be implemented for calling model forward function")

        loss = torch.mean(-torch.log(outputs.gather(dim=-1, index=targets.unsqueeze(-1))).squeeze(-1) * (targets!=pad_idx))
        valid_losses.append(loss.item())
    valid_loss = np.average(valid_losses)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'{OUTPUT_MODEL_PATH}/best_checkpoint.pt')
        
    print(f'Epoch={epoch+1}, updates={num_updates}, valid_loss={valid_loss}, best_valid_loss={best_valid_loss}')
    

print("Model training complete.")

