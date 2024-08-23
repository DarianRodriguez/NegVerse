# Negations-LM

This repository contains code for NegVerse: Generating Diverse Negations from Affirmative Sentences and its experiments. 

## Installation 
Run conda env create -f environment.yml && conda activate negation this will create the environment and install the required libraries. 

## Recreating NegVerse
Training: `python -m negator.PEFT.training`

Test a sentence: `python -m example.test`

Evaluation: `python -m example.evaluating`

The training configuration is loaded from `config.py`. Data should be placed in the `data/` folder. You can find the data structure [here](https://github.com/DarianRodriguez/Negations-LM/tree/main/negator/data). After training, the model is saved in the `negator/PEFT/peft_outputs` directory: [view the directory here](https://github.com/DarianRodriguez/Negations-LM/tree/main/negator/PEFT/peft_outputs).


The experiments reported use the following versions:

- **Transformers**: 4.42.4
- **PyTorch**: 2.3.1+cu118 (CUDA 11.8 support) 

The negation detection cue used in the filter is **NegBERT**. For details on the NegBERT model and its implementation, refer to the [official code](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation).

The trained model is provided in a `.zip` file, which can be downloaded from [this link](https://drive.google.com/file/d/1gmvAvaBC9ozqQdIBF1MOMLXdU3ljJrI_/view?usp=drive_link). After extracting the zip file, ensure that the contents are placed directly into the `NegBERT` folder located [here](https://github.com/DarianRodriguez/Negations-LM/tree/main/negator/selectors/NegBERT). The folder should contain the following files:

- `config.json`
- `pytorch_model.bin`

### Running Evaluation

To run the evaluation of the system, follow these steps:

1. **Install the Model:**
   - Install the trained models from the repository [here](https://github.com/kokeman/SOME).

2. **Place the Model in the Correct Folder:**
   - After installation, move the model files into the `gfm-models/` folder.
   - You can find the `gfm-models/` folder [here](https://github.com/DarianRodriguez/Negations-LM/tree/main/negator/Evaluation/gfm-models).

## Negation Demo
This code shows an example of how to use NegVerse:

```py
# initiate a wrapper.
from negator import Negator

# The affirmative sentence
text = "The kids were happy and loved the tea in London."

# Define instance of NegVerse
neg_verse = Negator()

# Subtree blanked sentences with is_token_only = False
blanks = negator_aug.get_random_blanked_sentences(text,max_blank_sent_count=6, max_blank_block = 2,is_token_only = False)
perturbations = neg_verse.perturb(text, blanked_sent = blanks, num_beams=5)
print(perturbations)

#return: ["The kids weren't happy and loved the tea in London.",
# 'None of the kids were happy and loved the tea in London.',
# "The kids weren't happy and loved the tea in London.",
# "The kids weren't happy and disgusted by the tea in London.",
# 'The kids were not happy and loved the tea in London.',
# 'The kids were not happy and disgusted by the tea in London.',
# 'The kids were unhappy and loved the tea in London.',
# 'None of them were happy and loved the tea in London.']

# token-only blanked sentences
blanks = negator_aug.get_random_blanked_sentences(text,max_blank_sent_count=6, max_blank_block = 2,is_token_only = True)
perturbations = neg_verse.perturb(text, blanked_sent = blanks, num_beams=5)
print(perturbations)

# return: [
#     'The kids were unhappy and hated the tea in London.',
#     "The kids weren't happy and disgusted by the tea in London.",
#     'The kids were unhappy and did not like the tea in London.',
#     'The kids were unhappy and avoided the tea in London.',
#     'The kids were not happy and loved the tea in London.',
#     'The kids were not happy and disgusted by the tea in London.',
#     'No kids were happy and loved the tea in London.',
#     'None of the kids were happy and loved the tea in London.',
#     "The kids weren't happy and loved the tea in London.",
#     'Not kids were happy and loved the tea in London.'
# ]

```