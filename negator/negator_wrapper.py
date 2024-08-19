import torch
import numpy as np
from spacy.tokens import Doc
from typing import List, Dict, Tuple
from .Evaluation.semantic_filter import load_distance_scorer
from .Evaluation import compute_edit_ops, compute_delta_perplexity,load_perplex_scorer,compute_sent_perplexity
from .generation_processing import get_outputs
from .PEFT.training_helper import setup_model
from peft import PeftModel
from .generation_processing import remove_blanks, Special_tokens
from .PEFT.inference import NegationModel
import warnings

from .helpers import create_processor

from .generation_processing import \
    get_random_idxes, \
    get_prompts,\
    create_blanked_sents

warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class Negator(object):
    def __init__(self, model_path: str="uw-hai/polyjuice", is_cuda: bool=True, directory: str = "negator/PEFT/peft_outputs") -> None:
        """The class to negate sentences
        Args:
            model_path (str, optional): The path to the generator.
                Defaults to "uw-hai/polyjuice". 
                No need to change this, unless you retrained a model
            is_cuda (bool, optional): whether to use cuda. Defaults to True.
        """
        # generator
        self.generator = None
        # validators
        self.polyjuice_generator_path = model_path
        self.perplex_scorer = None
        self.distance_scorer = None
        self.generator = None
        self.spacy_processor = None
        self.device, self.tokenizer, self.model = setup_model(model_path)
        self.loaded_model = PeftModel.from_pretrained(self.model, directory, is_trainable=False).to(self.device)

        self.is_cuda = is_cuda and torch.cuda.is_available()

    def validate_and_load_model(self, model_name: str) -> bool:
        """Validate whether the generator or scorer are loaded.
        If not, load the model.

        Args:
            model_name (str): the identifier of the loaded part.
                Should be [generator, perplex_scorer].

        Returns:
            bool: If the model is successfully load.
        """
        if getattr(self, model_name, None):
            return True
        else:
            loader = getattr(self, f"_load_{model_name}", None)
            return loader and loader()
        
    def _load_perplex_scorer(self):
        logger.info("Setup perplexity scorer.")
        self.perplex_scorer = load_perplex_scorer(is_cuda=self.is_cuda)
        return True   

    def _load_spacy_processor(self, is_space_tokenizer: bool=False):
        logger.info("Setup SpaCy processor.")
        self.spacy_processor = create_processor(is_space_tokenizer)
        return True

    def _process(self, sentence: str):
        if not self.validate_and_load_model("spacy_processor"): return None
        return self.spacy_processor(str(sentence))
    
    def _load_distance_scorer(self):
        self.distance_scorer = load_distance_scorer(self.is_cuda)
        return True
    
    def _compute_delta_perplexity(self, eops):
        if not self.validate_and_load_model("perplex_scorer"): return None
        return compute_delta_perplexity(eops, self.perplex_scorer, is_cuda=self.is_cuda,is_normalize=True)   
    


    ##############################################
    # validation
    ##############################################

    def get_random_blanked_sentences(self, 
        sentence: Tuple[str, Doc], 
        pre_selected_idxes: List[int]=None,
        deps: List[str]=None,
        is_token_only: bool=False,
        max_blank_sent_count: int=3,
        max_blank_block: int=1) -> List[str]:
        """Generate some random blanks for a given sentence

        Args:
            sentence (Tuple[str, Doc]): The sentence to be blanked,
                either in str or SpaCy doc.
            pre_selected_idxes (List[int], optional): 
                If set, only allow blanking a preset range of token indexes. 
                Defaults to None.
            deps (List[str], optional): 
                If set, only select from a subset of dep tags. Defaults to None.
            is_token_only (bool, optional):
                blank sub-spans or just single tokens. Defaults to False.
            max_blank_sent_count (int, optional): 
                maximum number of different blanked sentences. Defaults to 3.
            max_blank_block (int, optional): 
                maximum number of blanks per returned sentence. Defaults to 1.

        Returns:
            List[str]: blanked sentences
        """
        if type(sentence) == str:
            sentence = self._process(sentence)
            
        indexes = get_random_idxes(
            sentence, 
            pre_selected_idxes=pre_selected_idxes,
            deps=deps,
            is_token_only=is_token_only,
            max_count=max_blank_sent_count,
            max_blank_block=max_blank_block
        )

        blanked_sents = create_blanked_sents(sentence, indexes)
        return blanked_sents

    def generate_answers(
        self,
        prompts: List[str],
        num_beams: int = 3,
        num_return_sequences: int = 3) -> List[str]:
        """
        Generates multiple answer sequences from a list of prompts using beam search.

        Args:
            prompts (List[str]): A list of text prompts to generate answers for.
            num_beams (int, optional): The number of beams to use for beam search. Defaults to 3.
            num_return_sequences (int, optional): The number of sequences to return for each prompt. Defaults to 3.

        Returns:
            List[str]: A list of generated answers with blanks removed.
        """
    

        input_prompt = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs_peft_model = get_outputs(self.loaded_model, input_prompt, num_beams=num_beams, num_return_sequences=num_return_sequences)
        preds_list = self.tokenizer.batch_decode(outputs_peft_model, skip_special_tokens=True)

        preds_list_cleaned = []

        for sequence in preds_list:
            normalized, _ = remove_blanks(sequence)
            preds_list_cleaned.append(normalized)

        return preds_list_cleaned
    

    def perturb(self, 
        orig_sent: Tuple[str, Doc], 
        blanked_sent: Tuple[str, List[str]]=None,
        is_complete_blank: bool=False, 
        perplex_thred: int=10,
        num_perturbations: int=3,
        verbose: bool=False, 
        #is_include_metadata: bool=True,
        **kwargs) -> List[str]:
        """The primary perturbation function. Running example:
        Original sentence: 
            "It is great for kids."

        Args:
            orig_sent (Tuple[str, Doc]): 
                Original sentence, either in the form of str or SpaCy Doc.
            blanked_sents (Tuple[str, List[str]], optional): 
                sentences that contain blanks, e.g., "It is [BLANK] for kids."
                Defaults to None. If is None, the blank will be automatically placed.
                If is "" or incomplete form like "It is", set `is_complete_blank` to
                True below to allow the model to generate where to blank.
            is_complete_blank (bool, optional): 
                Whether the blanked sentence is already complete or not. 
                Defaults to False.
            ctrl_codes (Tuple[str, List[str]], optional): 
                The ctrl code (can be a list). Defaults to None. 
                If is None, will automatically become [resemantic, lexical,
                negation, insert, delete]. 
            perplex_thred (int, optional): 
                Perplexity filter for fluent perturbations. 
                we score both x and x' with GPT-2, and filter x' when the 
                log-probability (on the full sentence or the perturbed chunks) 
                decreases more than {perplex_thred} points relative to x
                Defaults to 5. If None, will skip filter.
            num_perturbations: 
                Num of max perturbations to collect. Defaults to 3.
            is_include_metadata: 
                Whether to return text, or also include other metadata and perplex score.
                Default to True.
            **kwargs: 
                The function can also take arguments for huggingface generators, 
                like top_p, num_beams, etc.
        Returns:
            List[str]: The perturbations.
        """

        orig_doc = self._process(orig_sent) if type(orig_sent) == str else orig_sent

        if blanked_sent:
            blanked_sents = [blanked_sent] if type(blanked_sent) == str else blanked_sent

        else:
            blanked_sents = self.get_random_blanked_sentences(orig_doc.text,max_blank_block=2)

        prompts = get_prompts(
            doc=orig_doc, 
            blanked_sents=blanked_sents, 
            is_complete_blank=is_complete_blank)       

        #print(prompts,"\n")

        generated = self.generate_answers(prompts,**kwargs) #num_beams=5

        validated_set = []
        for sentence in generated:
            # skip 
            if sentence in validated_set or sentence.lower() == orig_doc.text.lower(): 
                continue
            is_valid = True
            generated_doc = self._process(sentence)
            eop = compute_edit_ops(orig_doc, generated_doc)

            if perplex_thred is not None:
                pp = self._compute_delta_perplexity(eop)
                is_valid = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred


        return generated