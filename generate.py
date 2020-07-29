import numpy as np

def generate_letter(lm, history):
    """ Randomly picks letter according to probability distribution associated with
    the specified history, as stored in your language model.

    Note: returns dummy character "~" if history not found in model.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]]
        The n-gram language model.
        I.e. the dictionary: history -> [(char, freq), ...]

    history: str
        A string of length (n-1) to use as context/history for generating
        the next character.

    Returns
    -------
    str
        The predicted character. '~' if history is not in language model.
    """
    # STUDENT CODE HERE
    keys = []
    probs = []
    if history in lm:
        keys = list(zip(*lm[history]))
        #print(keys)
        return np.random.choice(keys[0], p=keys[1])
    else:
        return "~"

def generate_text(lm, n, nletters=100):
    """ Randomly generates `nletters` of text by drawing from
    the probability distributions stored in a n-gram language model
    `lm`.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]]
        The n-gram language model.
        I.e. the dictionary: history -> [(char, freq), ...]
    n: int
        Order of n-gram model.
    nletters: int
        Number of letters to randomly generate.

    Returns
    -------
    str
        Model-generated text.
    """
    # STUDENT CODE HERE
    string = ""
    history = "~"*(n-1)
    for i in range(nletters):
        string += generate_letter(lm, history)
        history = history[1:] + string[i]
    return string