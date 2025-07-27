import asyncio
from openai import AsyncOpenAI
import os
from transformers import AutoTokenizer

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")  # Set your API key in environment
)

# Load tokenizer
model_name = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the creative writing prompt
prompt = '''Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: The sun was shining brightly.

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.'''

prompt = "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$"

async def generate_response():
    # Create messages for chat completion
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template with reasoning disabled
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking/reasoning mode
    )
    
    # # Call Qwen3 model via OpenRouter
    # response = await client.chat.completions.create(
    #     model="Qwen/Qwen3-32B",  # Same model as in math_bench.py
    #     messages=messages,
    #     temperature=0.6,
    #     max_tokens=1000
    # )
    # 2️⃣ call the *completions* endpoint instead of the chat one
    response = await client.completions.create(        # ← was client.chat.completions.create
        model="Qwen/Qwen3-32B",
        prompt=text,                                   # ← was messages=messages
        temperature=0.6,
        max_tokens=1000
    )
    
    # Extract and return the response
    # return response.choices[0].message.content
    return response.choices[0].text

async def main():
    print("Calling Qwen3 via OpenRouter...")
    print("-" * 50)
    
    # Generate response
    response = await generate_response()
    
    # Print raw output
    print("Raw output:")
    print("-" * 50)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())