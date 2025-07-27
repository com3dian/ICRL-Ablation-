import asyncio
from openai import AsyncOpenAI
import os
from datasets import load_dataset

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")  # Set your API key in environment
)

async def generate_response(problem):
    # Create messages for chat completion
    messages = [
        {"role": "user", "content": problem}
    ]
    
    try:
        # Call Qwen3 model via OpenRouter
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-32B",  # Same model as in math_bench.py
            messages=messages,
            temperature=1.0,  # Same as math_bench.py
            max_tokens=4096   # Same as math_bench.py
        )
        
        # Extract and return the response
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

async def main():
    # Load AIME dataset
    print("Loading AIME dataset...")
    dataset = load_dataset("MathArena/aime_2025", split="train")
    
    # Get the first sample
    sample = dataset[0]
    problem = sample["problem"]
    answer = sample["answer"]
    
    print("Problem:")
    print("-" * 50)
    print(problem)
    print("\n" + "="*50)
    print("Reference Answer:", answer)
    print("="*50 + "\n")
    
    print("Calling Qwen3-32B via OpenRouter...")
    print("-" * 50)
    
    # Generate response
    response = await generate_response(problem)
    
    # Print raw output
    print("Model Output:")
    print("-" * 50)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())