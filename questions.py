import os
import string
from collections import Counter

import nltk
import sys
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_content = {}

    # Iterate over each file in directory and read the txt files
    for file in os.listdir(directory):
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(directory, file)) as f:
            file_content[file] = f.read()

    return file_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Set of punctuations and English stopwords
    invalid_entities = set(nltk.corpus.stopwords.words('english')).union(
        set(string.punctuation))

    return list(filter(lambda word: word not in invalid_entities,
                       nltk.word_tokenize(document.lower())))


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}

    # Counter object to map the words to their frequencies
    word_counts = Counter()

    # Populate word_counts using words in documents
    [word_counts.update(set(words)) for words in documents.values()]

    # Compute and store the idf for each word
    for word, frequency in word_counts.items():
        idfs[word] = math.log(len(documents) / frequency)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Return 'n' top files based on their tf-idf score
    return sorted(files.keys(),
                  key=lambda f: get_file_score(files[f], query, idfs),
                  reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Return 'n' top sentences based on their score and term frequency
    return sorted(sentences.keys(),
                  key=lambda s: (get_sentence_score(sentences[s], query, idfs),
                                 get_term_density(sentences[s], query)),
                  reverse=True)[:n]


def get_file_score(words_in_file, query, idfs):
    """
    Compute the tf-idf score for each word in the query and return the sum.
    """
    return sum(words_in_file.count(word) * idfs[word] for word in query)


def get_sentence_score(words_in_sentence, query, idfs):
    """
    Return the sum of idfs of each word that is both in the query and sentence.
    """
    return sum(idfs[word] for word in query if word in set(words_in_sentence))


def get_term_density(words_in_sentence, query):
    """
    Return the ratio of number of words that are both in the sentence and
    query to the number of words in the sentence.
    """
    return len([w for w in words_in_sentence if w in query]) / len(
        words_in_sentence)


if __name__ == "__main__":
    main()
