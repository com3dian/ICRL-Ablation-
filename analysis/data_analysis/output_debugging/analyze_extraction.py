import json
import re

# Load the output file
with open('/sfs/weka/scratch/ks8vf/code_submission/ICL/creative_writing_api/Qwen/Qwen3-32B/ICRL_new_eval_prompt_evalnum_100_n_20/output_list.json', 'r') as f:
    data = json.load(f)

# Pattern used in the code
pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

# Analyze a few cases
print("=== ANALYZING EXTRACTION FAILURES ===\n")

for sample_idx in range(min(5, len(data))):
    for round_idx in range(min(5, len(data[sample_idx]))):
        entry = data[sample_idx][round_idx]
        generated_text = entry['generated_text']
        answer = entry['answer']
        
        # Count answer tags
        open_count = generated_text.count('<answer>')
        close_count = generated_text.count('</answer>')
        
        if answer == "" or answer == "\n" or answer == "…":
            print(f"\nSample {sample_idx}, Round {round_idx}:")
            print(f"Answer extracted: {repr(answer)}")
            print(f"<answer> tags: {open_count}, </answer> tags: {close_count}")
            
            # Show portion of generated text around answer tags
            if '<answer>' in generated_text:
                start_idx = generated_text.find('<answer>')
                end_idx = generated_text.find('</answer>', start_idx) + len('</answer>')
                if end_idx > start_idx:
                    snippet = generated_text[max(0, start_idx-50):min(len(generated_text), end_idx+50)]
                    print(f"Text snippet: {repr(snippet[:200])}...")
                    
                    # Try to extract all answer blocks
                    all_matches = pattern.findall(generated_text)
                    print(f"All regex matches found: {len(all_matches)}")
                    if all_matches:
                        print(f"First match: {repr(all_matches[0][:100])}...")