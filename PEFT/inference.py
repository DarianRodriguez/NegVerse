
from training_helper import setup_model
from peft import PeftModel

def get_outputs(model,inputs,do_sample=True,num_beams=None,num_return_sequences = 3):
    """
    Generates multiple sequences of text using the provided model and inputs.

    Args:
        model: The model used for generation.
        inputs (dict): Input tensors including 'input_ids' and 'attention_mask'.
        do_sample (bool, optional): Whether to use sampling during generation (default: True).
        num_beams (int, optional): Number of beams for beam search. Overrides `do_sample`.
        num_return_sequences (int, optional): Number of sequences to generate per input (default: 3).

    Returns:
        torch.Tensor: Tensor containing generated sequences.
    """

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1000,
        early_stopping=False, #if num_beams is None else True, #The model can stop before reach the max_length
        temperature= 1,
        num_beams=1 if num_beams is None else num_beams,
        do_sample=num_beams is None and do_sample,
        num_return_sequences=num_return_sequences,
    )
    return outputs


class NegationModel:
    
    def __init__(self, output_directory: str, model_path: str = "uw-hai/polyjuice" ) -> None:
        """
        The wrapper class for inference with the specified model.

        Args:
            model_path (str): The path to the original model.
            output_directory (str): The path to the directory where the model is saved.
        """
        print(model_path)
        # Initialize tokenizer and model using setup_model function
        self.device, self.tokenizer, self.model = setup_model(model_path)

        # Load the pre-trained model from the specified output directory
        self.loaded_model = PeftModel.from_pretrained(self.model, output_directory, is_trainable=False)


    def infer(self, input_prompt_text: str, num_beams=5, num_return_sequences=3):
        """
        Perform inference with the loaded model and tokenizer.

        Args:
            input_prompt_text (str): The input prompt text to be tokenized and used for inference.
            num_beams (int): The number of beams for beam search. Default is 5.
            num_return_sequences (int): The number of sequences to generate per input. Default is 3.

        Returns:
            dict: Decoded outputs from the loaded model and the original model.
        """
        # Tokenize the input prompt
        input_prompt = self.tokenizer(input_prompt_text, return_tensors="pt")
        
        # Generate outputs using the loaded model
        outputs_loaded_model = get_outputs(self.loaded_model, input_prompt, num_beams=num_beams, num_return_sequences=num_return_sequences)
        
        # Generate outputs using the original model
        outputs_original_model = get_outputs(self.model, input_prompt, num_beams=num_beams, num_return_sequences=num_return_sequences)
        
        # Decode and return the outputs
        return {
            "trained_model_outputs": self.tokenizer.batch_decode(outputs_loaded_model, skip_special_tokens=True),
            "original_model_outputs": self.tokenizer.batch_decode(outputs_original_model, skip_special_tokens=True)
        }
       