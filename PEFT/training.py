
import config
import warnings
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_preprocess import TextGenerationSetup, process_dataframe, Trainer_preprocess
from training_helper import setup_model,create_model_directories, create_peft_model, train_engine
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model,PeftModel

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
print("Loading data")
affixal_df = '../data/affixal/filtered_df.pkl' # Specify the path to the pickle file
train_dataset = process_dataframe(affixal_df,text_format)
tokenized_datasets = train_dataset.map(text_format.tokenize_function)

# Load model setup
device,tokenizer, model = setup_model(config.MODEL_PATH)

# Prepare the dataset for training
print("Creating data loader")
prep_train = Trainer_preprocess(tokenizer, config.BATCH_SIZE)
train_dataloader, train_data, tok_data = prep_train.create_data_loader(tokenized_datasets)

# Create directory to save the trained model
working_dir="./"
output_dir_name="peft_outputs"
output_directory = create_model_directories(working_dir, output_dir_name)

# Training process
print("Training")

# Create new model for soft tuning 
model_peft = create_peft_model(config.NUM_VIRTUAL_TOKENS)
model_peft.to(device)

trained_model = train_engine(train_dataloader,config.NUM_EPOCHS, model_peft, device,config.NUM_VIRTUAL_TOKENS)

# Save the trained model
trained_model.save_pretrained(output_directory)
