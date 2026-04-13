# Used to clean text, remove punctuation
import re

# The common words that we want to ignore
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on",
    "for", "with", "as", "by", "at", "from", "that"
}

def preprocess(text):

    # Clean and prepare the text for analysis
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]

    # Return cleaned words
    return tokens

# Convert a natural language instruction into numeric parameters
def interpret_instruction(text):

    # Get cleaned words
    tokens = preprocess(text)

    # Default values for the agent behavior
    config = {
        "stability": 1.0,
        "smoothness": 0.0,
        "duration": 1.0
    }

    # Check each word to detect intention
    for word in tokens:

        # If the word is related to balance or stability
        if word in ["balance", "stability", "stable"]:
            config["stability"] += 0.5

        # If the word is related to smooth movement
        if word in ["minimize", "smooth", "gentle", "avoid", "sudden"]:
            config["smoothness"] += 0.5

        # If the word is related to time or duration
        if word in ["time", "long", "longest", "maintain", "maximize"]:
            config["duration"] += 0.5

    # Return the final configuration
    return config
