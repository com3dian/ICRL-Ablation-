import json
import re

# Load the output file
with open('/sfs/weka/scratch/ks8vf/code_submission/ICL/creative_writing_api/Qwen/Qwen3-32B/ICRL_new_eval_prompt_evalnum_100_n_20/output_list.json', 'r') as f:
    data = json.load(f)

# Look for cases where the generated text seems substantial but answer is empty/minimal
print("=== CASES WITH LIKELY GOOD RESPONSES BUT BAD EXTRACTION ===\n")

for sample_idx in range(min(20, len(data))):
    for round_idx in range(min(10, len(data[sample_idx]))):
        entry = data[sample_idx][round_idx]
        generated_text = entry['generated_text']
        answer = entry['answer']
        
        # Look for cases with substantial text but empty/minimal extraction
        if (answer in ["", "\n", "…", "\n\n"]) and len(generated_text) > 1000 and "Plan:" in generated_text and "Passage:" in generated_text:
            print(f"\nSample {sample_idx}, Round {round_idx}:")
            print(f"Answer extracted: {repr(answer)}")
            print(f"Generated text length: {len(generated_text)}")
            
            # Show a snippet that includes Plan and Passage
            plan_idx = generated_text.find("Plan:")
            if plan_idx != -1:
                snippet_start = max(0, plan_idx - 200)
                snippet_end = min(len(generated_text), plan_idx + 1000)
                print(f"\nSnippet around Plan/Passage:")
                print("-" * 80)
                print(generated_text[snippet_start:snippet_end])
                print("-" * 80)
                
                # Check if there's a proper answer structure
                if "<answer>" in generated_text and "Plan:" in generated_text and "Passage:" in generated_text:
                    # Find the last <answer> tag
                    all_answer_starts = [i for i in range(len(generated_text)) if generated_text.startswith('<answer>', i)]
                    if all_answer_starts:
                        last_answer_start = all_answer_starts[-1]
                        # Find corresponding </answer>
                        next_close = generated_text.find('</answer>', last_answer_start)
                        if next_close != -1:
                            actual_content = generated_text[last_answer_start+8:next_close]
                            if len(actual_content) > 100 and "Plan:" in actual_content:
                                print(f"\nFOUND LIKELY GOOD RESPONSE!")
                                print(f"Actual content length: {len(actual_content)}")
                                print(f"First 300 chars: {repr(actual_content[:300])}")
                                break
    else:
        continue
    break