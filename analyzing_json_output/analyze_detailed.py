import json
import re

# Load the output file
with open('/sfs/weka/scratch/ks8vf/code_submission/ICL/creative_writing_api/Qwen/Qwen3-32B/ICRL_new_eval_prompt_evalnum_100_n_20/output_list.json', 'r') as f:
    data = json.load(f)

# Pattern used in the code
pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

print("=== DETAILED ANALYSIS OF EXTRACTION FAILURES ===\n")

# Look for specific problematic patterns
problematic_cases = []

for sample_idx in range(min(10, len(data))):
    for round_idx in range(min(10, len(data[sample_idx]))):
        entry = data[sample_idx][round_idx]
        generated_text = entry['generated_text']
        answer = entry['answer']
        
        # Identify problematic extractions
        if answer in ["", "\n", "…", "\n\n"]:
            problematic_cases.append({
                'sample': sample_idx,
                'round': round_idx,
                'answer': answer,
                'generated_text': generated_text
            })

# Analyze the first few problematic cases in detail
for i, case in enumerate(problematic_cases[:5]):
    print(f"\n{'='*80}")
    print(f"CASE {i+1}: Sample {case['sample']}, Round {case['round']}")
    print(f"Extracted answer: {repr(case['answer'])}")
    print(f"{'='*80}")
    
    text = case['generated_text']
    
    # Find all answer tag positions
    answer_starts = []
    answer_ends = []
    
    pos = 0
    while True:
        start = text.find('<answer>', pos)
        if start == -1:
            break
        answer_starts.append(start)
        pos = start + 1
    
    pos = 0
    while True:
        end = text.find('</answer>', pos)
        if end == -1:
            break
        answer_ends.append(end)
        pos = end + 1
    
    print(f"\nFound {len(answer_starts)} <answer> tags at positions: {answer_starts[:5]}...")
    print(f"Found {len(answer_ends)} </answer> tags at positions: {answer_ends[:5]}...")
    
    # Show the context around the first answer tag
    if answer_starts:
        start = answer_starts[0]
        # Find the corresponding end tag
        end_tag_pos = -1
        for end in answer_ends:
            if end > start:
                end_tag_pos = end
                break
        
        if end_tag_pos != -1:
            # Show what's between the tags
            content_between = text[start+8:end_tag_pos]  # +8 for len('<answer>')
            print(f"\nContent between first <answer> and </answer>:")
            print(f"Length: {len(content_between)} characters")
            print(f"Content: {repr(content_between[:200])}")
            if len(content_between) > 200:
                print("... [truncated]")
            
            # Show surrounding context
            context_start = max(0, start - 100)
            context_end = min(len(text), end_tag_pos + 109)  # +9 for len('</answer>')
            print(f"\nContext around first answer tag:")
            print(repr(text[context_start:context_end]))
    
    # Check for common patterns that might cause issues
    print(f"\nPotential issues:")
    if '`<answer>…</answer>`' in text:
        print("- Found `<answer>…</answer>` in instructions (literal example)")
    if '<answer>\n</answer>' in text:
        print("- Found empty answer tags with just newline")
    if '</answer>' in text and '<answer>' not in text:
        print("- Found closing tag without opening tag")
    if text.count('<answer>') != text.count('</answer>'):
        print(f"- Mismatched tags: {text.count('<answer>')} opening vs {text.count('</answer>')} closing")