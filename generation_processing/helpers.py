

from PEFT.data_preprocess import Special_tokens

import re


def remove_blanks(total_sequence: str ) -> str:

    """
    Fills in the blanks in a template sentence with provided answers from the total sequence.

    The input string should have a special token '[NEG]' to mark the start of the relevant part,
    a '[SEP]' token to separate the template and answers, and each answer is marked by '[ANSWER]'.
    An '[EMPTY]' token is used to indicate an empty answer.

    Parameters:
    total_sequence (str): The input string containing the sequence with the template sentence and answers.

    Returns:
    tuple: A tuple containing the completed sentence and the list of answers.
    """
        
    try:
        text = total_sequence.split(Special_tokens.NEG_TOK)[-1]
        before, answers = text.split(Special_tokens.SEP_TOK)
        answers = [x.strip() for x in answers.split(Special_tokens.ANSWER_TOK)][:-1]
        answers = [x if x != Special_tokens.EMPTY_TOK else '' for x in answers]
        for a in answers:
            if a == '':
                before = re.sub(r' %s' % re.escape(Special_tokens.BLANK_TOK), a, before, count=1)
            else:
                before = re.sub(r'%s' % re.escape(Special_tokens.BLANK_TOK), a, before, count=1)
            final = before.split('!')[0] #remove pad token
        return final, answers
    except:
        return text, []





