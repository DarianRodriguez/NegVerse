import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from generator_helpers import remove_blanks, split_ctrl_code
import numpy as np
from compute_edit_ops import compute_edit_ops
from polyjuice import Polyjuice

class PerturbationGenerator:
    PERETURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    EMPTY_TOK = "[EMPTY]"
    ANSWER_TOK = "[ANSWER]"

    def __init__(self, model_path="uw-hai/polyjuice"):
        self.generator = self.setup_text_generation(model_path)
        self.pj = Polyjuice(model_path)
        
    def setup_text_generation(self, model_path):
        # Check if CUDA is available and set the device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move the model to the appropriate device
        model.to(device)
        
        # Set up the text generation pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="pt", device=device)
        
        return generator

    def batched_generate(self, examples, temperature=1, num_beams=None, num_return_sequences=3, do_sample=True, batch_size=128, **kwargs):
        preds_list = []
        with torch.no_grad():
            for e in range(0, len(examples), batch_size):
                preds_list += self.generator(
                    examples[e:e+batch_size],
                    temperature=temperature,
                    num_beams=1 if num_beams is None else num_beams,
                    max_length=1000,
                    early_stopping=False if num_beams is None else True,
                    do_sample=num_beams is None and do_sample,
                    num_return_sequences=num_return_sequences,
                    **kwargs
                )
        return preds_list

    def get_prompts(self, doc, blanked_sents, is_complete_blank=True):
        prompts = []
        for bt in blanked_sents:
            tag = 'negation'
            sep_tok = self.SEP_TOK if bt and is_complete_blank else ""
            prompts.append(f"{doc.strip()} {self.PERETURB_TOK} [{tag}] {bt.strip()}".strip())
        return prompts
    
    def get_prompt_context(self,text,blanks,examples):

        prompt_list = []

        inference_examples = self.get_prompts(text,blanks) # format the inference prompt

        for inference_example in inference_examples:
            # Construct the prompt with the examples and the inference example
            prompt_text = "\n".join(examples) + "\n" + inference_example
            prompt_list.append(prompt_text)

        return prompt_list

    def generate_on_prompts(self, prompts, temperature=1, num_beams=None, n=3, do_sample=True, batch_size=128, num_return_sequences=3):
        preds_list = self.batched_generate(prompts, temperature=temperature, num_beams=num_beams, do_sample=do_sample, batch_size=batch_size, num_return_sequences=num_return_sequences)
        print("HERE")
        print(preds_list)
        if len(prompts) == 1:
            preds_list = [preds_list]
        preds_list_cleaned = []
        for prompt, preds in zip(prompts, preds_list):
            prev_list = set()
            for s in preds:
                total_sequence = s["generated_text"].split(self.PERETURB_TOK)[-1]
                normalized, _ = remove_blanks(total_sequence)
                input_ctrl_code, normalized = split_ctrl_code(normalized)
                prev_list.add((input_ctrl_code, normalized))
            preds_list_cleaned.append(list(prev_list))
        return preds_list_cleaned

    def validate_and_sample_perturbations(self, generated, orig_doc, perplex_thred=None, num_perturbations=None):
        # Concatenate the generated outputs
        merged = list(np.concatenate(generated))
        orig_doc = self.pj._process(orig_doc)
        
        validated_set = []
        
        for _, gen in merged:
            # Skip if already in validated_set or is the same as the original document
            if gen in validated_set or gen.lower() == orig_doc.text.lower():
                continue
            
            is_valid = True
            generated_doc = self.pj._process(gen)
            eop = compute_edit_ops(orig_doc, generated_doc)
            
            if perplex_thred is not None:
                pp = self.pj._compute_delta_perplexity(eop)
                is_valid = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred
            
            if is_valid:
                ctrl = self.pj.detect_ctrl_code(orig_doc, generated_doc, eop)
                is_valid = is_valid and ctrl is not None and ctrl == "negation"
            
            if is_valid:
                validated_set.append(gen)
        
        if num_perturbations is None:
            num_perturbations = 1000
        
        # Sample the validated set
        sampled = np.random.choice(validated_set, min(num_perturbations, len(validated_set)), replace=False)
        
        return [str(s) for s in sampled]