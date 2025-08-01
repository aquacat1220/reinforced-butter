# type: ignore
"""
Simple Text Completion Script using Qwen with Chat Template
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
# model_name = "Qwen/Qwen3-0.6B"
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Your prompt
system_prompt = """You are a strategic assistant for the DEFENDER in a two-player capture game.
GAME RULES:
- 15x19 tile map with walls and walkable tiles
- Two flags: one REAL (attacker's target) and one DECOY
- ATTACKER: Knows which flag is real, must reach it before time limit
- DEFENDER: Cannot distinguish flags, must capture attacker by occupying same tile
- Movement: up/down/left/right or stay still (1 tile per turn)
- Defender starts closer to flags

STRATEGIC CONTEXT:
- Attacker will try to DECEIVE you about their true target
- Attacker may head toward decoy flag to mislead you
- Look for movement patterns that reveal true intentions:
* Direct/efficient routes vs deceptive detours
* Sudden direction changes or hesitation
* Time pressure affecting behavior
* Positioning relative to both flags

MAP FORMAT:
- 15 rows × 19 columns of comma-separated values
- NONE: walkable empty tile
- WALL: non-walkable tile  
- FLAG: flag position (you can't tell which is real)
- Coordinates given as (row, col) starting from (0,0) at top-left

YOUR TASK:
Analyze the movement history and predict the attacker's next k moves. Provide your answer as:
1. List of k coordinates: (row, col)
2. Brief reasoning explaining the pattern you detected

Focus on the most likely path - give your most confident prediction."""

user_prompt = """Static map layout:
WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL
WALL,NONE,NONE,NONE,WALL,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,WALL,NONE,NONE,NONE,WALL
WALL,NONE,WALL,NONE,WALL,NONE,WALL,WALL,WALL,WALL,WALL,WALL,WALL,NONE,WALL,NONE,WALL,NONE,WALL
WALL,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,WALL
WALL,NONE,WALL,NONE,WALL,NONE,WALL,WALL,WALL,WALL,WALL,WALL,WALL,NONE,WALL,NONE,WALL,NONE,WALL
WALL,FLAG,WALL,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,WALL,FLAG,WALL
WALL,NONE,WALL,WALL,WALL,WALL,NONE,WALL,WALL,WALL,WALL,WALL,NONE,WALL,WALL,WALL,WALL,NONE,WALL
WALL,NONE,NONE,NONE,NONE,NONE,NONE,WALL,WALL,WALL,WALL,WALL,NONE,NONE,NONE,NONE,NONE,NONE,WALL
WALL,NONE,WALL,WALL,WALL,WALL,NONE,WALL,WALL,WALL,WALL,WALL,NONE,WALL,WALL,WALL,WALL,NONE,WALL
WALL,NONE,WALL,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,WALL,NONE,WALL
WALL,NONE,WALL,NONE,WALL,NONE,WALL,WALL,WALL,WALL,WALL,WALL,WALL,NONE,WALL,NONE,WALL,NONE,WALL
WALL,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,WALL
WALL,NONE,WALL,NONE,WALL,NONE,WALL,WALL,WALL,WALL,WALL,WALL,WALL,NONE,WALL,NONE,WALL,NONE,WALL
WALL,NONE,NONE,NONE,WALL,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,NONE,WALL,NONE,NONE,NONE,WALL
WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL,WALL

Flag position:
Flag A: (5, 1)
Flag B: (5, 17)

Position history (past 1 moves):
Turn 1: Attacker(13,9), Defender(3,9)
...
Turn 2: Attacker(13,10), Defender(3,9)

Time remaining: 30 turns

Predict the attacker's next 4 moves:"""

# messages = [
#     {
#         "role": "system",
#         "content": system_prompt,
#     },
#     {
#         "role": "user",
#         "content": user_prompt,
#     },
# ]

# # Apply chat template
# text = tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
# )

# # Tokenize
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

model_inputs = tokenizer([system_prompt + "\n" + user_prompt], return_tensors="pt").to(
    model.device
)

# Generate text
generated_ids = model.generate(
    **model_inputs, max_new_tokens=1000, temperature=0.8, do_sample=True
)[0]

# Parse thinking content (Qwen-specific)
try:
    # Find the end of thinking section (token 151668 = </think>)
    index = len(generated_ids) - generated_ids[::-1].index(151668)
except ValueError:
    index = 0

# Decode thinking and final content separately
thinking_content = tokenizer.decode(
    generated_ids[:index], skip_special_tokens=True
).strip("\n")
final_content = tokenizer.decode(generated_ids[index:], skip_special_tokens=True).strip(
    "\n"
)

# Print results
if thinking_content:
    print("Thinking:", thinking_content)
    print("\nFinal answer:", final_content)
else:
    print("Generated text:", final_content)
