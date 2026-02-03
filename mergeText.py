import glob

# 1. Get a list of all your translated text files
# Assuming they are in a folder in Drive, e.g., 'translations'
file_paths = glob.glob(r"translations\*.txt") 

# 2. Merge them
output_file = "final_bangla_dapt_corpus.txt"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for fname in file_paths:
        with open(fname, 'r', encoding='utf-8') as infile:
            # Read the file and write it to the master file
            outfile.write(infile.read())
            # Add a newline just in case the file didn't end with one
            outfile.write("\n")

print(f"Merged {len(file_paths)} files into {output_file}")