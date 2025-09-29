import os 
import numpy as np
import torch
import torch.nn as nn 

from dataset import get_dataloader, alphabet
from model import ProGenForCausalLM

torch.manual_seed(0)

# Hyperparameters
BATCH_SIZE = 1

PRETRAINED_MODEL = "progen2-small"
PRETRAINED_MODEL_PATH = "pretrained_model"
DATA_PATH = "data"
OUTPUT_MODEL_PATH = "models"
OUTPUT_RESULT_PATH = "design"
OUTPUT_FILE_NAME = "design.txt"

# decoding parameters
MAX_SEQ_LENGTH = 512
DECODING_STRATEGY = "greedy" # you can also try topp

test_dataloader = get_dataloader(DATA_PATH, 'test', BATCH_SIZE, shuffle=False)
eos_idx = alphabet.eos()
fw = open(os.path.join(OUTPUT_RESULT_PATH, OUTPUT_FILE_NAME), 'w')

model = ProGenForCausalLM.from_pretrained(os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL)).to('cuda')
# TODO: loading model checkpoints
raise NotImplementedError("Need to be implemented for loading trained best model checkpoints")

print("\nStarting design...")
num_updates = 0
best_valid_loss = np.inf 
for prefix_seqs, _ in test_dataloader:
    prefix_seqs = prefix_seqs.to('cuda')
    # TODO: get model predicitons, calling model forward_inference
    raise NotImplementedError("Need to be implemented for model inference")

    designs = [alphabet.string(indexes[i]) for i in range(len(indexes))]
    print(designs[0])
    for design in designs:
        fw.write(design + '\n')
fw.close()
    
print("Inference complete.")

