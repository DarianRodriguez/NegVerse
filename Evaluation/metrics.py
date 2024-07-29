from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
from typing import List
import textdistance

def calculate_bleu(generated_texts: list, reference: str) -> float:
    """
    Calculate the average BLEU score for a list of generated texts against a single reference sentence.
    
    Args:
    - generated_texts (list): A list of generated text sentences.
    - reference (str): A single reference text sentence.
    
    Returns:
    - float: The average BLEU score of the generated texts against the reference.
   """
    
    reference_tokens = preprocess_text(reference)
    print(reference_tokens)
    bleu_scores = []

    # Preprocess both candidate and reference
    for candidate in generated_texts:
        if not isinstance(candidate, str):
            raise ValueError(f"Expected string for candidate, got {type(candidate).__name__}")
        
        candidate_tokens = preprocess_text(candidate)
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
        bleu_scores.append(bleu_score)

     # Calculate the average BLEU score
    if bleu_scores:
        return sum(bleu_scores) / len(bleu_scores)
    else:
        return 0.0

def levenshtein_distance(generated_texts: list, reference: str) -> float:
    """
    Calculate the average Levenshtein distance between a reference string and a list of generated texts.

    Parameters:
    generated_texts (list): List of strings to compare against the reference.
    reference (str): The reference string.

    Returns:
    float: The average Levenshtein distance.
    """

    total_distance = 0
    for candidate in generated_texts:
        distance = textdistance.levenshtein( reference, candidate)
        total_distance += distance
    average_distance = total_distance / len(generated_texts)

    # Compute Levenshtein distance
    return average_distance

def preprocess_text(text: str) -> List[str]:
    """
    Convert text to lowercase, tokenize, and remove punctuation.

    Args:
    - text (str): The input text to preprocess.

    Returns:
    - List[str]: A list of tokens (words) with punctuation removed.
    """
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation
    return tokens

