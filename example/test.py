
import pandas as pd
from negator import Negator
from negator.generation_processing import get_prompts
from negator.PEFT.data_preprocess import process_and_blank_sentences
from negator.generation_processing import get_outputs,remove_blanks

def generate_and_clean_predictions(negator_aug, blanks, get_outputs):

    device = negator_aug.device
    input_prompt = negator_aug.tokenizer(list(blanks), return_tensors='pt', padding=True, truncation=True).to(device)
    outputs_original_model = get_outputs(negator_aug.model, input_prompt, num_beams=5, num_return_sequences=3)
    preds_list = negator_aug.tokenizer.batch_decode(outputs_original_model, skip_special_tokens=True)

    preds_list_cleaned = []

    for sequence in preds_list:
        normalized, _ = remove_blanks(sequence)
        preds_list_cleaned.append(normalized)


    print(preds_list_cleaned)


negator_aug = Negator()

text = "The woman went to the gym"
print(text)

blanks = negator_aug.get_random_blanked_sentences(text,max_blank_sent_count=4, max_blank_block = 1,is_token_only = True)
print(blanks)


print("############################# MY model ##############")
perturbations = negator_aug.perturb(text, blanked_sent = blanks, num_beams=5)
#perturbations = negator_aug.perturb(text,num_beams=5)
print(perturbations)

#data_path = './negator/data/nli/RTE.txt'
#test_data  = process_and_blank_sentences(data_path, sample_size=1, max_blank=2,max_sent=3)
#print(test_data)

print("############################# Polyjuice #################")

generate_and_clean_predictions(negator_aug, blanks, get_outputs)