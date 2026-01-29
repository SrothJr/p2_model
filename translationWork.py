import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import os
import gc  # Garbage collection to free up VRAM
from tqdm.auto import tqdm

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
# Assuming you are running this LOCALLY on your laptop now:
# Update these paths to your actual local Windows paths
input_csv_path = r"C:\Users\ASUS\Desktop\Research_Material\p2_model\mental_health_post_clean_text.csv" 
output_txt_path = r"C:\Users\ASUS\Desktop\Research_Material\p2_model\bangla_corpus_for_dapt[7500-10000].txt"

START_ROW = 7900       
END_ROW = 10000      # Do small chunks (e.g., 2000 at a time) to be safe

# ==========================================
# 2. LOAD OPTIMIZED MODEL (The Fix)
# ==========================================
checkpoint = "facebook/nllb-200-distilled-600M"
print(f"Loading {checkpoint} in FP16...")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# OPTIMIZATION: torch_dtype=torch.float16
# This tells the RTX 3050 to use its Tensor Cores (Fast & Low Memory)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.float16 
).to("cuda")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    return text.strip()

def translate_batch_optimized(texts, batch_size=8):
    valid_texts = [t for t in texts if len(t) > 5]
    if not valid_texts: return []

    results = []
    bangla_lang_id = tokenizer.convert_tokens_to_ids("ben_Beng")

    # WRAP the range with tqdm to see a progress bar for every batch!
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Batch Progress", leave=False):
        sub_batch = valid_texts[i : i+batch_size]
        try:
            inputs = tokenizer(sub_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=bangla_lang_id,
                    max_length=512,
                    num_beams=3, 
                    early_stopping=True
                )

            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            results.extend(decoded)

            del inputs, translated_tokens
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
            
    return results

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
print(f"Starting Local Translation...")

file_mode = 'a' if os.path.exists(output_txt_path) else 'w'
chunk_size = 200 # Read smaller chunks for laptop safety
processed_count = 0
total_rows_to_read = END_ROW - START_ROW
skip_logic = range(1, START_ROW) if START_ROW > 0 else None

csv_iterator = pd.read_csv(
    input_csv_path,
    chunksize=chunk_size,
    skiprows=skip_logic,
    nrows=total_rows_to_read
)

with open(output_txt_path, file_mode, encoding="utf-8") as f_out:
    for chunk in csv_iterator:
        
        # Merge text columns
        if 'title' in chunk.columns and 'post' in chunk.columns:
            texts_to_translate = (chunk['title'].astype(str) + " " + chunk['post'].astype(str)).tolist()
        elif 'post' in chunk.columns:
            texts_to_translate = chunk['post'].astype(str).tolist()
        else:
            break

        cleaned_texts = [clean_text(t) for t in texts_to_translate]

        # Use batch_size=8 for RTX 3050 4GB
        bangla_texts = translate_batch_optimized(cleaned_texts, batch_size=8)

        for line in bangla_texts:
            f_out.write(line + "\n")

        processed_count += len(bangla_texts)
        print(f"Progress: {processed_count} / {total_rows_to_read} lines translated.")

print("Done.")