"""
python text_analysis.py --help

usage: text_analysis.py [-h] -f FILE [-m {nltk,spacy}] [-mw MAX_WORDS]
                        [-min MIN_FREQ] [-max MAX_FREQ] [-o OUTPUT]

Process text files and analyze word frequencies.

optional arguments:
  -h, --help            Show this help message and exit
  -f FILE, --file FILE  Path to the text file to process.
  -m {nltk,spacy}, --method {nltk,spacy}
                        Text processing method to use (default: spacy).
  -mw MAX_WORDS, --max-words MAX_WORDS
                        Maximum number of top words to display (default: 1000).
  -min MIN_FREQ, --min-freq MIN_FREQ
                        Minimum frequency for range filtering (default: 3).
  -max MAX_FREQ, --max-freq MAX_FREQ
                        Maximum frequency for range filtering (default: 4).
  -o OUTPUT, --output OUTPUT
                        Path to save the results to a file (optional).


# Run the script with a text file as input
python text_analysis.py --file path/to/your/textfile.txt

# Save the results to a file called "results.txt"
python text_analysis.py --file path/to/your/textfile.txt --output results.txt

# Specify the maximum number of top words
python text_analysis.py --file path/to/your/textfile.txt --max-words 500

# Specify a custom frequency range
python text_analysis.py --file path/to/your/textfile.txt --min-freq 5 --max-freq 10

# Combine multiple options
python text_analysis.py \
    --file path/to/your/textfile.txt \
    --method nltk \
    --max-words 200 \
    --min-freq 2 \
    --max-freq 8 \
    --output analysis.txt

"""
import os
import json
import argparse
from nltk.probability import FreqDist
import nltk
import spacy
from collections import defaultdict

from config import DEFAULT_REPO, DEFAULT_OUTPUT_REPO


# Utility Functions
def load_text(file_path):
    """Load and return text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def preprocess_with_nltk(text):
    """Tokenize and normalize text using NLTK."""
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    return [word.lower() for word in tokens if word.isalpha()]


def preprocess_with_spacy(text, spacy_model="de_core_news_sm", include_abstracts=True):
    """
    Preprocess text using SpaCy by lemmatizing words and optionally extracting abstracts.

    This function processes the input text with a specified SpaCy model to:
    1. Extract and lemmatize alphabetic words from the text.
    2. Optionally, if `include_abstracts=True`, identify sentences (abstracts) from the text
       containing each lemmatized word.

    Args:
        text (str): The input text to be processed.
        spacy_model (str): The SpaCy language model to use for processing.
            Default is "de_core_news_sm" (German small model).
        include_abstracts (bool): If `True`, extracts abstracts (sentences containing each word).
            If `False`, returns only the list of lemmatized words. Default is `False`.

    Returns:
        list: If `include_abstracts=False`, a list of lemmatized alphabetic words from the text.
        dict: If `include_abstracts=True`, a dictionary where keys are lemmatized words and
            values are lists of unique sentences (abstracts) containing those words.

    Examples:
        >>> text = "Das ist ein einfacher Text, um die Verarbeitung zu testen."
        >>> preprocess_with_spacy(text)
        ['das', 'sein', 'einfacher', 'text', 'um', 'verarbeitung', 'testen']

        >>> preprocess_with_spacy(text, include_abstracts=True)
        {
            'verarbeitung': ['Das ist ein einfacher Text, um die Verarbeitung zu testen.'],
            'text': ['Das ist ein einfacher Text, um die Verarbeitung zu testen.'],
            'sein': ['Das ist ein einfacher Text, um die Verarbeitung zu testen.'],
            ...
        }
    """
    nlp = spacy.load(spacy_model)
    doc = nlp(text)

    # List to store lemmatized words
    lemmatized_words = [token.lemma_.lower() for token in doc if token.is_alpha]

    # Dictionary to store words and their abstracts
    word_abstracts = defaultdict(list)

    if include_abstracts:
        # Iterate through sentences in the text
        for sent in doc.sents:
            sent_text = sent.text
            for token in sent:
                if token.is_alpha:  # Keep only alphabetic tokens
                    lemma = token.lemma_.lower()
                    # Add the sentence as an abstract if it contains the word
                    word_abstracts[lemma].append(sent_text)

        # Ensure each word has unique abstracts
        word_abstracts = {word: list(set(abstracts)) for word, abstracts in word_abstracts.items()}

    return lemmatized_words, word_abstracts


def get_highest_frequency(freq_dist):
    """Return the highest frequency in the distribution."""
    return max(freq_dist.values(), default=0)


def get_top_words(freq_dist, max_words):
    """Return words sorted by descending frequency, up to max_words."""
    return freq_dist.most_common(max_words)


def get_words_in_range(freq_dist, min_freq, max_freq):
    """Return words with frequencies between min_freq and max_freq."""
    return [(word, freq) for word, freq in freq_dist.items() if min_freq <= freq <= max_freq]


def resolve_file_path(file_arg, default_repo=DEFAULT_REPO):
    """
    Resolve the full path for the file argument.
    If the file path is not absolute or relative, prepend the default repository.
    """
    if not os.path.isabs(file_arg) and not os.path.exists(file_arg):
        return os.path.join(default_repo, file_arg)
    return file_arg


def save_results_to_file_(file_path, top_words, words_in_range):
    """Save the results to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Top Words by Frequency:\n")
        for word, freq in top_words:
            file.write(f"{word}: {freq}\n")

        file.write("\nWords in Frequency Range:\n")
        for word, freq in words_in_range:
            file.write(f"{word}: {freq}\n")


def save_results_to_files(top_words, words_in_range, word_abstracts, params, output_dir=DEFAULT_OUTPUT_REPO):
    """
    Save the results to two separate files with auto-generated names.

    Args:
        output_dir (str): Directory where the output files will be saved.
        top_words (list of tuples): List of (word, frequency) for the top words.
        word_abstracts (dict): Dictionary of lemmatized words and their abstracts.
        words_in_range (list of tuples): List of (word, frequency) for words in the frequency range.
        params (dict): Dictionary containing parameters for file naming. Keys include:
                    - 'max_words' (int): Maximum number of top words to include.
                    - 'min_freq' (int): Minimum frequency for the words in the range.
                    - 'max_freq' (int): Maximum frequency for the words in the range.

    Returns:
        tuple: Paths to the two generated files (top_words_file, range_words_file).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate file names
    top_words_file = os.path.join(output_dir, f"top_{params['max_words']}_words.txt")
    range_words_file = os.path.join(output_dir, f"words_in_range_{params['min_freq']}_{params['max_freq']}.txt")
    abstracts_file = os.path.join(output_dir, f"abstracts_words_in_range_{params['min_freq']}_{params['max_freq']}.json")

    # Save top words by frequency to the first file
    with open(top_words_file, 'w', encoding='utf-8') as file:
        file.write("Top Words by Frequency:\n")
        for word, freq in top_words:
            file.write(f"{word}: {freq}\n")

    # Save words in frequency range to the second file
    with open(range_words_file, 'w', encoding='utf-8') as file:
        file.write("Words in Frequency Range:\n")
        for word, freq in words_in_range:
            file.write(f"{word}: {freq}\n")

    # Save word abstracts to a JSON file
    with open(abstracts_file, 'w', encoding='utf-8') as file:
        json.dump(word_abstracts, file, ensure_ascii=False, indent=4)

    return top_words_file, range_words_file, abstracts_file


# Main Execution as CLI Application
def main():
    parser = argparse.ArgumentParser(description="Process text files and analyze word frequencies.")
    parser.add_argument(
        "-f", "--file", required=True, help="Path to the text file to process (default directory: /data)."
    )
    parser.add_argument(
        "-m", "--method", choices=["nltk", "spacy"], default="spacy",
        help="Text processing method to use (default: spacy)."
    )
    parser.add_argument(
        "-mw", "--max-words", type=int, default=1000,
        help="Maximum number of top words to display (default: 1000)."
    )
    parser.add_argument(
        "-min", "--min-freq", type=int, default=3,
        help="Minimum frequency for range filtering (default: 3)."
    )
    parser.add_argument(
        "-max", "--max-freq", type=int, default=4,
        help="Maximum frequency for range filtering (default: 4)."
    )
    args = parser.parse_args()

    # Resolve the file path
    file_path = resolve_file_path(args.file)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    # Load Text
    text = load_text(file_path)

    # Choose Processing Method
    if args.method == "nltk":
        print("Using NLTK for text processing...")
        nltk.download("punkt")
        processed_words = preprocess_with_nltk(text)
    elif args.method == "spacy":
        print("Using SpaCy for text processing...")
        processed_words, word_abstracts = preprocess_with_spacy(text)

    # Calculate Frequency Distribution
    freq_dist = FreqDist(processed_words)

    # # Analysis
    # print(f"\n--- Analysis Results ---")
    # print(f"Highest frequency: {get_highest_frequency(freq_dist)}")

    top_words = get_top_words(freq_dist, args.max_words)
    words_in_range = get_words_in_range(freq_dist, args.min_freq, args.max_freq)
    abstracts_words_in_range = {_w: word_abstracts[_w] for _w, _f in words_in_range}

    params = {'max_words': args.max_words, 'min_freq': args.min_freq, 'max_freq': args.max_freq}

    # Save Results to File
    save_results_to_files(top_words, words_in_range, abstracts_words_in_range, params)
    print(f"\nResults saved to {DEFAULT_OUTPUT_REPO}")

if __name__ == "__main__":
    main()
