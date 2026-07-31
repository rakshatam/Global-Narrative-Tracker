import os
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Define explicit local cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "ml_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Device Configuration: Use CUDA if available
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

print(f"ML Engine initializing. Using device: {DEVICE_STR}")
print(f"Model cache directory: {CACHE_DIR}")

# 1. Zero-Shot Stance Classifier
# We use a lightweight model suitable for 4GB VRAM
STANCE_MODEL_NAME = "facebook/bart-large-mnli"

print(f"Loading Stance Classifier: {STANCE_MODEL_NAME}...")
# This will download the model to CACHE_DIR on the first run, and load from cache subsequently.
stance_tokenizer = AutoTokenizer.from_pretrained(STANCE_MODEL_NAME, cache_dir=CACHE_DIR)
stance_model = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL_NAME, cache_dir=CACHE_DIR)

stance_classifier = pipeline(
    "zero-shot-classification",
    model=stance_model,
    tokenizer=stance_tokenizer,
    device=DEVICE
)
print("Stance Classifier loaded.")

# 2. Embedding Model for Cython LSH Clustering
# We use all-MiniLM-L6-v2 which is extremely small and fast
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=CACHE_DIR)
if DEVICE_STR == "cuda":
    embedding_model = embedding_model.to("cuda")
print("Embedding Model loaded.")

def get_stance(text: str, topic: str):
    """
    Given a text and a topic, predict the stance using the zero-shot classifier.
    Labels: positive, negative, neutral
    """
    candidate_labels = ["positive stance", "negative stance", "neutral stance"]
    # The pipeline handles moving tensors to the configured device
    result = stance_classifier(text, candidate_labels, multi_label=False)
    
    # Return the highest scoring label and its score
    top_label = result['labels'][0].replace(" stance", "")
    score = result['scores'][0]
    return {"stance": top_label, "confidence": float(score)}

from transformers import AutoModelForSeq2SeqLM

# 3. Summarization Model
# We use distilbart-cnn-12-6 because it is distilled (lighter) and fits well within a 4GB VRAM limit
SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
print(f"Loading Summarization Model: {SUMMARIZATION_MODEL_NAME}...")

# Bypassing the buggy pipeline registry entirely and loading the raw model tensors directly
summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME, cache_dir=CACHE_DIR)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZATION_MODEL_NAME, 
    cache_dir=CACHE_DIR, 
    use_safetensors=True
)

if DEVICE_STR == "cuda":
    summary_model = summary_model.to("cuda")

print("Summarization Model loaded.")

def summarize_text(text: str):
    """
    Summarizes a full scraped article down to a concise paragraph using manual generation.
    Handles long texts by truncating to model limits.
    """
    try:
        # brutally slice the string to ~800 words to avoid crashing the tokenizer limit (1024 tokens)
        safe_text = " ".join(text.split()[:800])
        
        # Tokenize and move to GPU if available
        inputs = summary_tokenizer(safe_text, return_tensors="pt", max_length=1024, truncation=True)
        if DEVICE_STR == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Generate summary tensors
        summary_ids = summary_model.generate(
            inputs["input_ids"], 
            max_length=130, 
            min_length=30, 
            num_beams=4,
            early_stopping=True
        )
        
        # Decode back to english string
        summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
            
    except Exception as e:
        print(f"Summarization error: {e}")
        # Fallback to returning the first 500 characters if ML fails
        return text[:500] + "..."

def get_embedding(text: str):
    """
    Generate an embedding vector for the text using the fast MiniLM model.
    Used for the LSH Cython engine.
    """
    # Returns numpy array
    return embedding_model.encode(text)

if __name__ == "__main__":
    # Test the models
    topic = "Global Water Scarcity Initiative"
    text = "The head of state announced a brilliant new plan to combat water shortages, which is highly promising."
    
    print("Testing Stance Detection...")
    stance_result = get_stance(text, topic)
    print(f"Stance Result: {stance_result}")
    
    print("Testing Embedding Generation...")
    emb = get_embedding(text)
    print(f"Embedding shape: {emb.shape}")
