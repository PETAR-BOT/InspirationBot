from collections import Counter
import numpy as np

from collections import defaultdict

def normalize(counter):
    """ Convert a `letter -> count` counter to a list
   of (letter, frequency) pairs, sorted in descending order of
   frequency.

    Parameters
    -----------
    counter : collections.Counter
        letter -> count

    Returns
    -------
    List[Tuple[str, int]]
       A list of tuples - (letter, frequency) pairs in order
       of descending-frequency

    Examples
    --------
    letter_count = Counter({"a": 1, "b": 3})
    letter_count
    Counter({'a': 1, 'b': 3})

    normalize(letter_count)
    [('b', 0.75), ('a', 0.25)]
    """
    # STUDENT CODE HERE
    counter = counter.most_common()
    temp_freq = []
    total = 0
    for i in range(len(counter)):
        letter, num = enumerate(counter[i])
        # print(letter, num)
        temp_freq.append((letter[1], num[1]))
        total += num[1]
    freq = [(temp_freq[i][0], temp_freq[i][1] / total) for i in range(len(temp_freq))]
    return freq




def train_lm(text, n):
    """ Train character-based n-gram language model.

    This will learn: given a sequence of n-1 characters, what the probability
    distribution is for the n-th character in the sequence.

    For example if we train on the text:
        text = "cacao"

    Using a n-gram size of n=3, then the following dict would be returned.
    See that we *normalize* each of the counts for a given history

        {'ac': [('a', 1.0)],
         'ca': [('c', 0.5), ('o', 0.5)],
         '~c': [('a', 1.0)],
         '~~': [('c', 1.0)]}

    Tildas ("~") are used for padding the history when necessary, so that it's
    possible to estimate the probability of a seeing a character when there
    aren't (n - 1) previous characters of history available.

    So, according to this text we trained on, if you see the sequence 'ac',
    our model predicts that the next character should be 'a' 100% of the time.

    For generating the padding, recall that Python allows you to generate
    repeated sequences easily:
       `"p" * 4` returns `"pppp"`

    Parameters
    -----------
    text: str
        A string (doesn't need to be lowercased).
    n: int
        The length of n-gram to analyze.

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
      {n-1 history -> [(letter, normalized count), ...]}
    A dict that maps histories (strings of length (n-1)) to lists of (char, prob)
    pairs, where prob is the probability (i.e frequency) of char appearing after
    that specific history.

    Examples
    --------
    train_lm("cacao", 3)
    {'ac': [('a', 1.0)],
     'ca': [('c', 0.5), ('o', 0.5)],
     '~c': [('a', 1.0)],
     '~~': [('c', 1.0)]}
    """
    # STUDENT CODE HERE
    # prefix = Counter(text)
    history = "~" * (n - 1)
    strings = defaultdict(Counter)
    for char in text:
        strings[history][char] += 1
        history = history[1:] + char
    final_dict = {history: normalize(count) for history, count in strings.items()}
    return final_dict
