from ..PEFT import config
from ..PEFT.inference import NegationModel
from ..negator_wrapper import Negator
from typing import List
from torch.utils.data import DataLoader
from ..generation_processing import remove_blanks
from ..PEFT.data_preprocess import Special_tokens
from .eval_metrics import compute_closeness,compute_self_bleu

import numpy as np

def evaluate_model(test_data:list):

    print("\n Evaluating the model")
    directory = "negator/PEFT/peft_outputs"
    negation_model = NegationModel(directory)
    negator_object = Negator()

    #create batches
    test_dataloader = DataLoader(test_data, batch_size=8, drop_last=False)

    generations_original = []
    generations_peft = []
    org_tree_dist, org_lev_dist = [], []
    peft_tree_dist, peft_lev_dist = [], []
    peft_bleu, org_bleu = [], []

    # Query the model
    for batch in test_dataloader:
        outputs = negation_model.infer(batch, num_beams=5, num_return_sequences=3)
        generations_peft.extend(outputs["trained_model_outputs"])
        generations_original.extend(outputs["original_model_outputs"])

    # Process the generated text: remove blanks, lowercase, 
    for index in range(len(generations_original)):
        replaced_peft, _ = remove_blanks(generations_peft[index])
        replaced_original, _ = remove_blanks(generations_original[index])
        original = generations_peft[index].split(Special_tokens.PERETURB_TOK)[0]

        # Convert text to doc
        ref_doc = negator_object._process(original.lower())
        gen_doc = negator_object._process(replaced_peft.lower())
        org_doc = negator_object._process(replaced_original.lower())

        # Compute closeness and diversity metric with trained model
        closeness = compute_closeness([gen_doc],ref_doc)
        peft_tree_dist.append(closeness['tree_dist'])
        peft_lev_dist.append(closeness['edit_dist'])

        # Compute closeness and diversity metric with original model
        closeness = compute_closeness([org_doc],ref_doc)
        org_tree_dist.append(closeness['tree_dist'])
        org_lev_dist.append(closeness['edit_dist'])

        diversity = compute_self_bleu([gen_doc], ref_doc)
        peft_bleu.append(diversity['bleu4'])
        diversity = compute_self_bleu([org_doc], ref_doc)
        org_bleu.append(diversity['bleu4'])


    avg_tree_peft = np.mean(peft_tree_dist)
    avg_lev_peft = np.mean(peft_lev_dist)
    avg_tree_orig = np.mean(org_tree_dist)
    avg_lev_orig = np.mean(org_lev_dist)
    avg_bleu_orig = np.mean(org_bleu)
    avg_bleu_peft = np.mean(org_lev_dist)

    # Format the averages for printing
    print("Closeness")
    print(f"Average Syntactic Tree Distance (PEFT): {avg_tree_peft:.3f}")
    print(f"Average Levenshtein Distance (PEFT): {avg_lev_peft:.3f}")
    print(f"Average Syntactic Tree Distance (Original): {avg_tree_orig:.3f}")
    print(f"Average Levenshtein Distance (Original): {avg_lev_orig:.3f}")

    print("\nDiversity")
    print(f"Average Bleu Score (Original): {avg_bleu_orig:.3f}")
    print(f"Average Bleu Score (PEFT): {avg_bleu_peft:.3f}")

    return {
    'avg_tree_peft': avg_tree_peft,
    'avg_lev_peft': avg_lev_peft,
    'avg_tree_orig': avg_tree_orig,
    'avg_lev_orig': avg_lev_orig,
    'avg_bleu_orig': avg_bleu_orig,
    'avg_bleu_peft': avg_bleu_peft 
    }

    





    
