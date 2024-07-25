
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
from inference import get_outputs, NegationModel
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
new_affixal_path = '../data/affixal/generated_sentences.txt' 
new_affixal_df = process_data(new_affixal_path) # Create dataset with MASK

train_dataset_affixal_1 = process_dataframe(affixal_df,text_format,True)
train_dataset_affixal_2 = process_dataframe(affixal_df,text_format, False) # entire sentence
train_dataset_affixal_3 = process_dataframe_general(new_affixal_df ,text_format,True)
train_dataset_affixal_4 = process_dataframe_general(new_affixal_df ,text_format,False)
train_dataset_affixal = concatenate_datasets([train_dataset_affixal_1, train_dataset_affixal_2,train_dataset_affixal_3,train_dataset_affixal_4])

# Load non verbal negations
data_path = '../data/non_verbal/sentence_negated_modified.txt' 
nonverbal_df = process_data(data_path)  # Create dataset with MASK

train_dataset_nonverbal_1 = process_dataframe_general(nonverbal_df,text_format,True)
train_dataset_nonverbal_2 = process_dataframe_general(nonverbal_df,text_format,False)
train_dataset_nonverbal = concatenate_datasets([train_dataset_nonverbal_1, train_dataset_nonverbal_2])

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

# Save the trained model
trained_model.save_pretrained(output_directory)

# Inference
input_prompt = "His behavior is always responsible. <|perturb|> [negation] [BLANK]"

negation_model = NegationModel(output_directory)
outputs = negation_model.infer(input_prompt, num_beams=5, num_return_sequences=3)

print("Trained Model Outputs:")
print(outputs["trained_model_outputs"])
print("\nOriginal Model Outputs:")
print(outputs["original_model_outputs"])


