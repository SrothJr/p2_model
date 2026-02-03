import re
from tqdm.auto import tqdm

# CONFIGURATION
INPUT_FILE = "final_bangla_dapt_corpus.txt"  # Your merged file path
OUTPUT_FILE = "final_bangla_dapt_corpus_CLEANED.txt"      # Where to save the clean version

def is_valid_line(text):
    if not isinstance(text, str) or not text.strip():
        return False

    # 1. Remove URLs (MIRoBERTa [7])
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 2. Check Language Ratio (Remove chunks that are mostly English)
    # Count Bangla characters (Unicode range 0980-09FF)
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # If the line has very little Bangla or is mostly English, discard it.
    if bangla_chars < 3: return False 
    if english_chars > bangla_chars: return False

    # 3. Check for Word Count (MIRoBERTa [1])
    # "All data instances were restricted to... a word count of at least ten words"
    words = text.split()
    if len(words) < 8: # Using 8 as a safe buffer
        return False

    # 4. Check for Hallucination/Loops (The "Drug Drug Drug" problem in Source [4])
    # If unique words are less than 25% of total words, it's a loop.
    if len(words) > 20: # Only check ratios for longer sentences
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25: 
            return False

    # 5. Check for Punctuation Spam (Source [6] "?????")
    # If >40% of the line is punctuation/symbols, discard.
    total_len = len(text)
    punct_len = len(re.findall(r'[^\w\s\u0980-\u09FF]', text)) # Non-word, non-space, non-Bangla
    if (punct_len / total_len) > 0.4:
        return False

    return True

# EXECUTION
print("Starting cleaning process...")
clean_lines = []
total_lines = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        total_lines += 1
        # Normalize whitespace
        clean_line = ' '.join(line.split())
        
        if is_valid_line(clean_line):
            clean_lines.append(clean_line + "\n")

# SAVE
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.writelines(clean_lines)

print(f"\nProcessing Complete!")
print(f"Original Lines: {total_lines}")
print(f"Cleaned Lines:  {len(clean_lines)}")
print(f"Discarded:      {total_lines - len(clean_lines)}")
print(f"Saved to:       {OUTPUT_FILE}")