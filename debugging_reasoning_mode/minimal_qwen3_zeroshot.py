from vllm import LLM, SamplingParams

# Initialize vLLM with Qwen3
llm = LLM(model="Qwen/Qwen3-32B", 
          tensor_parallel_size=1,
          gpu_memory_utilization=0.95)

# Define the prompt for creative writing
prompt = '''Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: The sun was shining brightly.

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.'''

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=1000
)

# Generate response
outputs = llm.generate([prompt], sampling_params)

# Print raw output
print("Raw output:")
print("-" * 50)
print(outputs[0])