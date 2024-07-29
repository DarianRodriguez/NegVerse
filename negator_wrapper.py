import torch
import numpy as np
from spacy.tokens import Doc
from typing import List, Dict, Tuple
from Evaluation.semantic_filter import load_distance_scorer, compute_sent_cosine_distance
from PEFT.training_helper import setup_model
from peft import PeftModel

from helpers import create_processor


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class Negator(object):
    def __init__(self, model_path: str="uw-hai/polyjuice", is_cuda: bool=True, directory: str = "PEFT/peft_outputs") -> None:
        """The wrapper.

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
        self.loaded_model = PeftModel.from_pretrained(self.model, directory, is_trainable=False)

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

    ##############################################
    # validation
    ##############################################
    

    def perturb(self, 
        orig_sent: Tuple[str, Doc], 
        blanked_sent: Tuple[str, List[str]]=None,
        is_complete_blank: bool=False, 
        ctrl_code: Tuple[str, List[str]]=None, 
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
