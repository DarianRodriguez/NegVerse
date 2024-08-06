
import pandas as pd
from negator import Negator
from negator.PEFT.data_preprocess import process_and_blank_sentences
from negator.Evaluation.evaluate import evaluate_model

data_path = './negator/data/nli/SNLI.txt'
test_data  = process_and_blank_sentences(data_path, sample_size=2, max_blank=2,max_sent=4, is_token_only= True)
print("\nModel with token only")
evaluate_model(test_data, file_name="token")

test_data  = process_and_blank_sentences(data_path, sample_size=2, max_blank=2,max_sent=4, is_token_only= False)
print("\nModel with subtree")
evaluate_model(test_data, file_name="subtree")