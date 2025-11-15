import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import string
import argparse
from nltk import word_tokenize
import nltk
from nltk.corpus import wordnet, stopwords
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# when you design a function, you should think about 
# what are input and expected output 

# stop_words = set(stopwords.words('english'))

# def introducing_typo(word):
#     """ """
#     ## word is a string 
#     if len(word) <= 3:
#         return word 

#     typo_choice = random.choice(['delete', 'swap'])

#     if typo_choice == 'delete':
#         idx = random.randint(0, len(word) -1)
#         # [] is left inclusive and right exclusive 
#         return word[:idx] + word[idx+1:]
#     elif typo_choice == 'swap': 
#         idx = random.randint(0, len(word) -2)
#         return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    
#     return word 



# def custom_transform(example):
#     ################################
#     ##### YOUR CODE BEGINGS HERE ###

#     # Design and implement the transformation as mentioned in pdf
#     # You are free to implement any transformation but the comments at the top roughly describe
#     # how you could implement two of them --- synonym replacement and typos.

#     # You should update example["text"] using your transformation

#     # synonym + typo 
#     # what i want to achieve is to ranomely replace one word with its synonym 


#     text = example['text']
#     words = text.split()

#     # create a candidate indices first 
#     # candidate are words which have synonyms 
#     candidate_indices = [i for i, w in enumerate(words)
#     if w.lower() not in stop_words and wordnet.synsets(w.strip(string.punctuation))]

#     # randomely pick one word from the candidte list to replace

#     if candidate_indices:
#         num_changes = min(5, len(candidate_indices))  # up to 2 changes per sentence
#         for i in random.sample(candidate_indices, num_changes):
#             word = words[i].strip(string.punctuation)
#             syns = wordnet.synsets(word)

#             if syns:  # make sure the word actually has synonyms
#                 lemmas = [
#                     l.name().replace("_", " ")
#                     for l in syns[0].lemmas()
#                     if l.name().lower() != word.lower()
#                 ]

#                 # randomly pick one word from the synonym list
#                 if lemmas:
#                     words[i] = random.choice(lemmas)

#     # === typo introduction 
#     # it requires the knowledge of stop_words to be able to know w.lower() is needed 
#     typo_candidates = [i for i, w in enumerate(words) if w.lower() not in stop_words and w.isalpha()]
    
#     if typo_candidates and random.random() < 0.7:
#         j = random.choice(typo_candidates)
#         words[j] = introducing_typo(words[j])

    
#     delete_candidates = [
#     i for i, w in enumerate(words)
#     if w.lower() not in stop_words and len(w) > 3
# ]

#     if delete_candidates:
#         num_deletes = min(2, len(delete_candidates))
#         for i in sorted(random.sample(delete_candidates, num_deletes), reverse=True):
#             del words[i]

#     example["text"] = " ".join(words)

#     ##### YOUR CODE ENDS HERE ######

#     if random.random() < 0.05:  
#         print("=" * 60)
#         print("Original:", text)
#         print("Transformed:", " ".join(words))
#         print("=" * 60)


#     return example


# ===== testing ======
# import random
# import string
# from nltk.corpus import wordnet, stopwords

stop_words = set(stopwords.words("english"))

# helper: mild typos
def introducing_typo(word):
    if len(word) <= 3:
        return word
    typo_choice = random.choice(["delete", "swap"])
    if typo_choice == "delete":
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1:]
    elif typo_choice == "swap":
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    return word

# confusable characters
UNICODE_MAP = {
    "a": "à", "e": "é", "i": "í", "o": "ò", "u": "ù",
    "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"
}

HOMOPHONES = {"too": "2", "to": "2", "for": "4", "you": "u", "are": "r", "before": "b4", "see": "c", "okay": "ok", "love": "luv"}
DISTRACTORS = [
    "btw I saw the trailer first", "no spoilers tho", "my cousin fell asleep halfway",
    "I read it on Reddit lol", "streamed it on Netflix actually", "ngl I was on my phone half the time",
    "idk maybe I was tired", "honestly my cat was louder than the movie", "tbh not even sure why I watched it"
]
POSITIVE_EMOJIS = ["😊", "❤️", "😄", "👍", "🎬"]
NEGATIVE_EMOJIS = ["😒", "💀", "👎", "😞", "😡"]

def custom_transform(example):
    text = example["text"]
    words = text.split()

    # 1️⃣ Shuffle sentences
    sentences = text.split(". ")
    if len(sentences) > 2 and random.random() < 0.8:
        random.shuffle(sentences)
        text = ". ".join(sentences)
        words = text.split()

    # 2️⃣ Replace some words with homophones or unicode confusables
    new_words = []
    for w in words:
        if random.random() < 0.2:
            w = HOMOPHONES.get(w.lower(), w)
        if random.random() < 0.15 and any(ch in UNICODE_MAP for ch in w):
            w = "".join(UNICODE_MAP.get(ch, ch) for ch in w)
        new_words.append(w)
    words = new_words

    # 3️⃣ Inject negation confusion
    if random.random() < 0.4:
        text = " ".join(words)
        text = text.replace(" good ", " not good ").replace(" bad ", " not bad ")
        words = text.split()

    # 4️⃣ Insert random distractor clause
    if random.random() < 0.7:
        insert_pos = random.randint(1, len(words) - 2)
        words.insert(insert_pos, random.choice(DISTRACTORS))

    # 5️⃣ Add typos or delete short stopwords
    typo_candidates = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    for i in random.sample(typo_candidates, k=min(3, len(typo_candidates))):
        if random.random() < 0.6:
            words[i] = introducing_typo(words[i])
    words = [w for w in words if not (w.lower() in stop_words and random.random() < 0.2)]

    # 6️⃣ Random case noise and punctuation flooding
    words = [
        w.upper() if random.random() < 0.15 else
        w.lower() if random.random() < 0.1 else w
        for w in words
    ]
    if random.random() < 0.5:
        text = " ".join(words).replace(".", "!!!").replace(",", " ,,, ")
    else:
        text = " ".join(words)

    # 7️⃣ Emoji irony
    lower_text = text.lower()
    if any(word in lower_text for word in ["good", "great", "love", "amazing", "excellent", "wonderful"]):
        emoji = random.choice(NEGATIVE_EMOJIS)
    elif any(word in lower_text for word in ["bad", "boring", "hate", "terrible", "awful", "disappointing"]):
        emoji = random.choice(POSITIVE_EMOJIS)
    else:
        emoji = random.choice(POSITIVE_EMOJIS + NEGATIVE_EMOJIS)
    if random.random() < 0.8:
        if random.random() < 0.5:
            text = emoji + " " + text
        else:
            text = text + " " + emoji

    # 8️⃣ Final assembly
    example["text"] = text

    if random.random() < 0.03:
        print("=" * 80)
        print("Original:", example["text"][:200], "...")
        print("=" * 80)

    return example


   
