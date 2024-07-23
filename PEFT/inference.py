


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