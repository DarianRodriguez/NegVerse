import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import Dataset
from enum import Enum
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from datasets import Dataset
from difflib import SequenceMatcher


class TargetType(Enum):
  PAD = 0
  CONTEXT = 1
  CONTEXT_SPECIAL = 2
  CONTEXT_INFILL_SEP = 3
  INFILL = 4
  INFILL_SPECIAL = 5

class Special_tokens():
    PERETURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    ANSWER_TOK = "[ANSWER]"
    NEG_TOK = "[negation]"

    @classmethod
    def initialize_token_ids(cls, tokenizer):
        cls.PERETURB_TOK_ID = [tokenizer(cls.PERETURB_TOK)['input_ids'], tokenizer(" " + cls.PERETURB_TOK)['input_ids']]
        cls.BLANK_TOK_ID = [tokenizer(cls.BLANK_TOK)['input_ids'], tokenizer(" " + cls.BLANK_TOK)['input_ids']]
        cls.SEP_TOK_ID = [tokenizer(cls.SEP_TOK)['input_ids'], tokenizer(" " + cls.SEP_TOK)['input_ids']]
        cls.ANSWER_TOK_ID = [tokenizer(cls.ANSWER_TOK)['input_ids'], tokenizer(" " + cls.ANSWER_TOK)['input_ids']]
        cls.NEG_TOK_ID = [tokenizer(cls.NEG_TOK)['input_ids'], tokenizer(" " + cls.NEG_TOK)['input_ids']]


class TextGenerationSetup:

    """
    Class used to setup the prompts format for training
    """
    
    PERETURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    ANSWER_TOK = "[ANSWER]"

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = None #self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        # Add a new pad token if it doesn't exist and set it to ID 0
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = TargetType.PAD.value


    def get_prompts(self,doc, blanked_sents, is_complete_blank=True):
        prompts = []
        for bt in blanked_sents:
            tag = 'negation'
            sep_tok = TextGenerationSetup.SEP_TOK if bt and is_complete_blank else ""
            new_prompt = f"{doc.strip()} {TextGenerationSetup.PERETURB_TOK} [{tag}] {bt.strip()}".strip()
            #prompts.append(new_prompt)
            prompts.append(new_prompt.rstrip('.').strip())
        return prompts

    def get_answer_2(self,answer):
        prompts = []
        prompts.append(f"{TextGenerationSetup.SEP_TOK} {answer.strip()} {TextGenerationSetup.ANSWER_TOK}")
        #prompts.append(answer.strip())
        return prompts
    
    def get_answer(self,answers):
        sentence = ""
        initial = TextGenerationSetup.SEP_TOK
        for answer in answers:
            sentence += f"{initial} {answer.strip()} {TextGenerationSetup.ANSWER_TOK}"
            initial = ""
        return sentence
    
    
    def tokenize_function(self, examples):

        input_encodings = self.tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=100)
        return input_encodings
    
def process_dataframe(affixal_path, text_format ):

        """
        Processes a DataFrame containing text data to generate a dataset suitable for text generation tasks.

        This function reads a DataFrame from a pickle file, processes each row to replace specific cues with 
        a blank token, generates prompts and answers in the required format, and converts the processed data 
        into a format suitable for training a Hugging Face model.

        Args:
            affixal_path (str): Path to the pickle file containing the DataFrame with text data.
            text_format (TextGenerationSetup): An instance of the TextGenerationSetup class used for formatting prompts and answers.

        Returns:
            Dataset: A Hugging Face Dataset object containing the processed input and target texts.
        """

        train_data = []
        sentence_mask = True

        # Load the DataFrame from the pickle file
        filtered_df = pd.read_pickle(affixal_path)

        for _, row in filtered_df.iterrows():
            text = row['text']
            text_pos = row['text_substituted']
            cue = row['cues'].split()[0]  # Assuming 'cues' column contains space-separated cues
            
            if sentence_mask :
                # Replace the cue in the text with '[BLANK]'
                text_with_blank = text.replace(cue, '[BLANK]')

                # Generate the prompt and answer
                prompt_examples = text_format.get_prompts(text_pos, [text_with_blank]) # format the input prompts
                answer_formatted = text_format.get_answer([cue]) # format the answer

            else:
                text_with_blank = '[BLANK]'
                prompt_examples = text_format.get_prompts(text_pos, [text_with_blank]) # format the input prompts
                answer_formatted = text_format.get_answer([text]) # format the answer

            
            # Combine the prompt and answer in the required format
            combined_sentence = f"{prompt_examples[0]} {answer_formatted}"
            train_data.append(combined_sentence)

        # Convert to a suitable format for Hugging Face Dataset
        train_dataset= pd.DataFrame(train_data, columns=["input_text"]) #columns=["input_text", "target_text"]
        train_dataset = Dataset.from_pandas(train_dataset)
        
        return train_dataset

def process_dataframe_general(filtered_df,text_format):

        """
        Processes a DataFrame containing text data to generate a dataset suitable for text generation tasks.
        this dataset have the negation, original snetence, the masked information
        """

        train_data = []
        sentence_mask = True

        for _, row in filtered_df.iterrows():
            negated = row['negated']
            original = row['original']
            
            
            if sentence_mask :
                # Replace the cue in the text with '[BLANK]'
                text_with_blank = row['mask']
                answer = row['masked_info']
            else:
                text_with_blank = '[BLANK]'
                answer = [negated]
            
            # Generate the prompt and answer
            prompt_examples = text_format.get_prompts(original, [text_with_blank]) # format the input prompts
            answer_formatted = text_format.get_answer(answer) # format the answer
            
            # Combine the prompt and answer in the required format
            combined_sentence = f"{prompt_examples[0]} {answer_formatted}"
            train_data.append(combined_sentence)

        # Convert to a suitable format for Hugging Face Dataset
        train_dataset= pd.DataFrame(train_data, columns=["input_text"]) #columns=["input_text", "target_text"]
        train_dataset = Dataset.from_pandas(train_dataset)
        
        return train_dataset

def mask_differences(negated, original):

    # Tokenize sentences
    negated_tokens = np.array(negated.split())
    original_tokens = np.array(original.split())
    
    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, negated_tokens, original_tokens)
    
    # Initialize mask tokens and list to capture masked segments
    mask_tokens = []
    masked_info = []
    prev_end = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete':
            if prev_end < i1:
                # Add [BLANK] for non-matching segments
                mask_tokens.extend(['[BLANK]'] * (i1 - prev_end))
                # Capture the masked segment
                masked_info.append(' '.join(negated_tokens[prev_end:i1]))
            else:
                # Consolidate contiguous [BLANK] tokens
                if mask_tokens and mask_tokens[-1] == '[BLANK]':
                    mask_tokens = mask_tokens[:-1]
                mask_tokens.append('[BLANK]')
                masked_info.append(' '.join(negated_tokens[i1:i2]))
        elif tag == 'insert':
            # Skip inserted parts as they are not in negated_tokens
            pass
        else:  # 'equal'
            # Add matching tokens to the mask
            mask_tokens.extend(negated_tokens[i1:i2])
        
        prev_end = i2
    
    # Handle trailing [BLANK] segments
    if prev_end < len(negated_tokens):
        mask_tokens.extend(['[BLANK]'] * (len(negated_tokens) - prev_end))
        masked_info.append(' '.join(negated_tokens[prev_end:]))
    
    # Consolidate contiguous [BLANK] segments into a single [BLANK]
    mask_array = []
    in_blank_segment = False
    for token in mask_tokens:
        if token == '[BLANK]':
            if not in_blank_segment:
                mask_array.append('[BLANK]')
                in_blank_segment = True
        else:
            mask_array.append(token)
            in_blank_segment = False
    
    # Convert mask tokens to a single string
    masked_sentence = ' '.join(mask_array)
    
    return masked_sentence, masked_info

def process_data(data_path):

    df_dataset = pd.read_csv(data_path, delimiter='\t')

    # Apply function to the DataFrame
    df_dataset[['mask', 'masked_info']] = df_dataset.apply(lambda row: pd.Series(mask_differences(row['negated'], row['original'])), axis=1)

    # Convert 'masked_info' column to lists of masked segments
    df_dataset['masked_info'] = df_dataset['masked_info'].apply(lambda x: [seg for seg in x if seg.strip()])

    return df_dataset


class Trainer_preprocess:
    def __init__(self, tokenizer, batch_size = 8):
        self.tokenizer = tokenizer
        #self.train_dataset = train_dataset
        self.batch_size = batch_size

    def tokenized_special_tokens(self):
        # Call the function to initialize token IDs
        Special_tokens.initialize_token_ids(self.tokenizer)

    def extract_token(self, token_list, special_tok_ids, target_value):
        # Mask the position of special tokens
            token_len = len(special_tok_ids[0])
            index = 0

            if any(token_list[:token_len] == tok_id for tok_id in special_tok_ids):
                label = [target_value] * token_len
                index = token_len  # Skip the next three tokens as they are part of the negation marker
            else:
                label = None

            return index, label
    
    
    def align_labels(self, input_data):
        tokens = input_data['input_ids']

        # Initialize the list of labels
        labels = [TargetType.PAD.value] * len(tokens)

        special_list = [
            Special_tokens.PERETURB_TOK_ID, 
            Special_tokens.BLANK_TOK_ID, 
            Special_tokens.NEG_TOK_ID,
            Special_tokens.SEP_TOK_ID,
            Special_tokens.ANSWER_TOK_ID
        ]
        target_list = [
            TargetType.CONTEXT_SPECIAL.value,
            TargetType.CONTEXT_SPECIAL.value,
            TargetType.CONTEXT_SPECIAL.value,
            TargetType.CONTEXT_INFILL_SEP.value,
            TargetType.INFILL_SPECIAL.value
        ]

        # Assign labels
        i = 0
        while i < len(tokens) and tokens[i] != TargetType.PAD.value:
            token_processed = False
            for special_token, target_type in zip(special_list, target_list):
                step, label = self.extract_token(tokens[i:], special_token, target_type)
                if step != 0 and label is not None:
                    token_processed = True
                    break

            if token_processed:
                labels[i:i + step] = label
                i += step
            else:
                i += 1

        # Modify the labels list to add context and answer mask
        labels = [
            1 if x == TargetType.PAD.value and 3 in labels[i:] else  # All zeros before 2 should be changed to 1
            4 if x == TargetType.PAD.value and 3 in labels[:i] and 5 in labels[i:] else  # All zero values between 3 and 5 should be changed to 4
            x
            for i, x in enumerate(labels)
        ]

        return {'aligned_labels': labels}
    
    def create_data_loader(self,tokenized_dataset):
        # Call tokenized_special_tokens to initialize token IDs
        self.tokenized_special_tokens()

        # Apply align_labels function to the dataset
        train_dataset = tokenized_dataset.map(self.align_labels)

        # Extract input_ids and aligned_labels
        input_ids = np.array(train_dataset['input_ids'])
        aligned_labels = np.array(train_dataset['aligned_labels'])

        # Convert to PyTorch tensors
        input_ids_tensor = torch.from_numpy(input_ids.astype(np.int64))
        aligned_labels_tensor = torch.from_numpy(aligned_labels.astype(np.int64))

        # Create TensorDataset
        train_data = TensorDataset(input_ids_tensor, aligned_labels_tensor)

        # Create RandomSampler for training
        train_sampler = RandomSampler(train_data)

        # Create DataLoader
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, drop_last=True)

        return train_dataloader, train_data, train_dataset