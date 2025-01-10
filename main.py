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
import argparse
from nltk.probability import FreqDist
import nltk
import spacy

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


def preprocess_with_spacy(text, spacy_model="de_core_news_sm"):
    """Lemmatize text using SpaCy."""
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]


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


def resolve_output_path(output_arg, default_repo=DEFAULT_OUTPUT_REPO):
    """
    Resolve the full path for the output file.
    If the path is not absolute, prepend the default output repository.
    Ensure the directory exists.
    """
    # Ensure the output directory exists
    if not os.path.exists(default_repo):
        os.makedirs(default_repo)

    if not os.path.isabs(output_arg):
        return os.path.join(default_repo, output_arg)
    return output_arg


def save_results_to_file(file_path, top_words, words_in_range):
    """Save the results to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Top Words by Frequency:\n")
        for word, freq in top_words:
            file.write(f"{word}: {freq}\n")

        file.write("\nWords in Frequency Range:\n")
        for word, freq in words_in_range:
            file.write(f"{word}: {freq}\n")


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
    parser.add_argument(
        "-o", "--output", default="results.txt",
        help="Path to save the results to a file (default directory: /out)."
    )
    args = parser.parse_args()

    # Resolve the file path
    file_path = resolve_file_path(args.file)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    output_path = resolve_output_path(args.output)

    # Load Text
    text = load_text(file_path)

    # Choose Processing Method
    if args.method == "nltk":
        print("Using NLTK for text processing...")
        nltk.download("punkt")
        processed_words = preprocess_with_nltk(text)
    elif args.method == "spacy":
        print("Using SpaCy for text processing...")
        processed_words = preprocess_with_spacy(text)

    # Calculate Frequency Distribution
    freq_dist = FreqDist(processed_words)

    # Analysis
    print(f"\n--- Analysis Results ---")
    print(f"Highest frequency: {get_highest_frequency(freq_dist)}")

    top_words = get_top_words(freq_dist, args.max_words)
    print(f"Top {args.max_words} words by frequency:")
    for word, freq in top_words:
        print(f"{word}: {freq}")

    words_in_range = get_words_in_range(freq_dist, args.min_freq, args.max_freq)
    print(f"\nWords with frequencies between {args.min_freq} and {args.max_freq}:")
    for word, freq in words_in_range:
        print(f"{word}: {freq}")

    # Save Results to File
    save_results_to_file(output_path, top_words, words_in_range)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
