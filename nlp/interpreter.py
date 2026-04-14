# Used to clean text, remove punctuation
import re

# Common words that we want to ignore
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on",
    "for", "with", "as", "by", "at", "from", "that"
}

def preprocess(text):

    # Clean and prepare the text for analysis
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]

    # Return cleaned words
    return tokens


# Convert a natural language instruction into numeric parameters
def interpret_instruction(text):

    # Get unique cleaned words (avoid duplicates influence)
    tokens = set(preprocess(text))

    # Default values for the agent behavior
    config = {
        "stability": 1.0,
        "smoothness": 0.0,
        "duration": 1.0
    }

    # Check each word to detect intention
    for word in tokens:

        # Words related to balance or stability (higher importance)
        if word in ["balance", "stability", "stable", "equilibrium"]:
            config["stability"] += 1.0

        # Words related to smooth movement (basic smoothness)
        if word in ["minimize", "smooth", "gentle"]:
            config["smoothness"] += 0.5

        # Words that emphasize avoiding abrupt changes (stronger penalty)
        if word in ["avoid", "sudden"]:
            config["smoothness"] += 0.7

        # Words related to duration or time (medium importance)
        if word in ["time", "long", "longest"]:
            config["duration"] += 0.7

        # Words that emphasize maintaining or maximizing performance
        if word in ["maintain", "maximize"]:
            config["duration"] += 0.5

    # Normalize values to prevent them from growing too large
    for key in config:
        config[key] = min(config[key], 2.0)

    # Return the final configuration
    return config