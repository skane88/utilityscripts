"""
File to generate wordlists for use with a diceware style passphrase generator

Wordlists taken from: http://www.ashley-bovan.co.uk/words/partsofspeech.html
"""

import random
from itertools import product
from pathlib import Path


def get_nouns(no_rolls=5):

    no_choices = 6**no_rolls

    noun_base = (
        Path("..") / Path("data") / Path("parts_of_speech_word_files") / Path("nouns")
    )

    nouns = []

    for f in ["1syllablenouns.txt", "2syllablenouns.txt"]:

        word_file = noun_base / Path(f)

        with open(word_file, "r") as file_handle:

            words = file_handle.read().splitlines()

        words = [w.strip() for w in words if w != ""]
        nouns += words

    words = random.sample(population=nouns, k=no_choices)
    numbers = product("123456", repeat=no_rolls)

    return zip(numbers, words)


def get_adjs(no_rolls=5):
    no_choices = 6**no_rolls

    adj_base = (
        Path("..")
        / Path("data")
        / Path("parts_of_speech_word_files")
        / Path("adjectives")
    )

    adjs = []

    for f in [
        "1syllableadjectives.txt",
        "2syllableadjectives.txt",
        "3syllableadjectives.txt",
    ]:
        word_file = adj_base / Path(f)

        with open(word_file, "r") as file_handle:
            words = file_handle.read().splitlines()

        words = [w.strip() for w in words if w != ""]
        adjs += words

    words = random.sample(population=adjs, k=no_choices)
    numbers = product("123456", repeat=no_rolls)

    return zip(numbers, words)


def make_lists():

    nouns = get_nouns()
    adjs = get_adjs()

    noun_file = Path("nouns.txt")
    adj_file = Path("adjs.txt")

    with open(noun_file, "w") as f:
        for n in nouns:
            f.write(f"{''.join(n[0])}: {n[1]}\n")

    with open(adj_file, "w") as f:
        for a in adjs:
            f.write(f"{''.join(a[0])}: {a[1]}\n")


if __name__ == "__main__":

    make_lists()
