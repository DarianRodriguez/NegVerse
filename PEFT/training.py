
import config
import warnings
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_preprocess import TextGenerationSetup, process_dataframe, Trainer_preprocess, process_dataframe_general, process_data
from training_helper import setup_model,create_model_directories, create_peft_model, train_engine
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model,PeftModel
from inference import get_outputs
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets

def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

# Suppress all warnings
warnings.filterwarnings("ignore")

set_random_seed(config.SEED)

# Setup the model
text_format = TextGenerationSetup(config.MODEL_PATH)

# Load affixal negations dataset in the right format for training
print("\n Loading data")
affixal_df = '../data/affixal/filtered_df.pkl' # Specify the path to the pickle file
train_dataset_affixal = process_dataframe(affixal_df,text_format)

# Load non verbal negations
data_path = '../data/non_verbal/sentence_negated_modified.txt' 
nonverbal_df = process_data(data_path)
train_dataset_nonverbal = process_dataframe_general(nonverbal_df,text_format)

# Unify datasets
unified_dataset = concatenate_datasets([train_dataset_affixal, train_dataset_nonverbal])

tokenized_datasets = unified_dataset.map(text_format.tokenize_function)

# Load model setup
device,tokenizer, model = setup_model(config.MODEL_PATH)

# Prepare the dataset for training
print("\n Creating data loader")
prep_train = Trainer_preprocess(tokenizer, config.BATCH_SIZE)
train_dataloader, train_data, tok_data = prep_train.create_data_loader(tokenized_datasets)

# Create directory to save the trained model
working_dir="./"
output_dir_name="peft_outputs"
output_directory = create_model_directories(working_dir, output_dir_name)

# Training process
print("\n Training")

# Create new model for soft tuning 
model_peft = create_peft_model(config.NUM_VIRTUAL_TOKENS)
model_peft.to(device)

trained_model = train_engine(train_dataloader,config.NUM_EPOCHS, model_peft, device,config.NUM_VIRTUAL_TOKENS)

#count_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of trainable parameters
#print(f'Number of trainable parameters: {count_trainable_params}')

## Inference
#model.eval()f
#input_prompt = tokenizer("His behavior is always responsible. <|perturb|> [negation] His behavior is [BLANK]", return_tensors="pt").to(device)
#input_prompt = tokenizer("too much of it feels focused and underdeveloped . <|perturb|> [negation] too much of it feels [BLANK] and underdeveloped", return_tensors="pt").to(device)
#outputs_prompt = get_outputs(trained_model, input_prompt, num_beams = 5)
#print(tokenizer.batch_decode(outputs_prompt, skip_special_tokens=True))


# Save the trained model
trained_model.save_pretrained(output_directory)


# Inference
input_prompt = tokenizer("His behavior is always responsible. <|perturb|> [negation] [BLANK]", return_tensors="pt") #.to(device)

loaded_model = PeftModel.from_pretrained(model,output_directory,is_trainable=False)
outputs_prompt1 = get_outputs(loaded_model , input_prompt, num_beams = 5)
outputs_prompt2 = get_outputs(model , input_prompt, num_beams = 5)
print(tokenizer.batch_decode(outputs_prompt1, skip_special_tokens=True))
print("ORIGINAL")
print(tokenizer.batch_decode(outputs_prompt2, skip_special_tokens=True))


