from openai import OpenAI
import time


def extract_reasoning_and_prediction(text):
    if "REASONING:" in text and "PREDICTION:" in text:
        reasoning_part = text.split("REASONING:")[1].split("PREDICTION:")[0].strip()
        prediction_part = text.split("PREDICTION:")[1].strip()

        # Manually parse tuples from lines like "(13, 10)"
        predictions = []
        for line in prediction_part.splitlines():
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                # Remove parentheses and split by comma
                coords = line[1:-1].split(",")
                if len(coords) == 2:
                    try:
                        row = int(coords[0].strip())
                        col = int(coords[1].strip())
                        predictions.append((row, col))
                    except ValueError:
                        continue  # Skip if not valid integers

        return reasoning_part, predictions
    else:
        return None, None


client = OpenAI()

system_prompt = """You are a strategic assistant for the DEFENDER in a two-player capture game.

GAME RULES:
- Game is played on a 15x19 grid.
- Each tile is either:
  - "WALL": not walkable
  - "NONE": walkable
  - "FLAG": walkable
- There are two FLAGs, one flag is a DECOY, and the other is the REAL
- Players can move up/down/left/right or stay in place (1 tile per turn)
- Only "NONE" and "FLAG" tiles are walkable
- Players cannot move outside the 15x19 boundaries

ROLES:
- ATTACKER: Knows which FLAG is REAL, and tries to reach it before the time limit
- DEFENDER: Doesn't know which FLAG is REAL, and must stop the attacker from reaching it
- DEFENDER can capture the ATTACKER by landing on the same tile
- Defender starts closer to the flags but cannot tell which flag is real

ATTACKER STRATEGY:
- Attacker may try to **deceive** the defender by moving toward the decoy
- Look for strategic clues:
  - Direct paths vs. deceptive detours
  - Sudden direction changes or hesitations
  - Position relative to both flags
  - Time pressure

MAP FORMAT:
- Sent as JSON with:
  - "walkable": list of (row, col) coordinates that are walkable (either "NONE" or "FLAG")
  - "flags": list of two (row, col) positions
  - "attacker": current attacker position as (row, col)
  - "defender": current defender position as (row, col)
  - "history": list of previous turns, each turn a dict with "attacker" and "defender" positions
  - "time_remaining": number of turns left

YOUR TASK:
1. Analyze movement history to detect deceptive vs direct behavior
2. Predict the attacker's next k moves by:
   2-1. Identify all adjacent walkable tiles
   2-2. Use movement pattern and flag positions to choose the most likely next move
   2-3. Update attacker's position
3. Final validation:
   - All moves are 1-tile or stationary
   - All moves are inside map boundaries
   - All moves are on "walkable" tiles

OUTPUT FORMAT:
First, explain your logic under the REASONING section.
Then, output the list of predicted coordinates under the PREDICTION section.

Example:

REASONING:
[Your reasoning here...]

PREDICTION:
(13, 11)
(13, 12)
(12, 12)
(11, 12)
"""

user_prompt = """{
    "walkable": [[1, 1], [1, 2], [1, 3], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 15], [1, 16], [1, 17],
        [2, 1], [2, 3], [2, 5], [2, 13], [2, 15], [2, 17],
        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 16], [3, 17],
        [4, 1], [4, 3], [4, 5], [4, 13], [4, 15], [4, 17],
        [5, 1], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 17],
        [6, 1], [6, 6], [6, 12], [6, 17],
        [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17],
        [8, 1], [8, 6], [8, 12], [8, 17],
        [9, 1], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [9, 17],
        [10, 1], [10, 3], [10, 5], [10, 13], [10, 15], [10, 17],
        [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17],
        [12, 1], [12, 3], [12, 5], [12, 13], [12, 15], [12, 17],
        [13, 1], [13, 2], [13, 3], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 15], [13, 16], [13, 17]
    ],
    "flags": [[5, 1], [5, 17]],
    "attacker": [13, 10],
    "defender": [3, 9],
    "history": [
        {"attacker": [13, 9], "defender": [3, 9]},
        {"attacker": [13, 10], "defender": [3, 9]}
    ],
    "time_remaining": 30
}


Predict the attacker's next 4 moves:"""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": user_prompt,
    },
]

start = time.time()
print(start)

response = client.responses.create(model="gpt-4.1-mini", input=messages, temperature=0)

end = time.time()
print(end)

print(end - start)

reasoning, predictions = extract_reasoning_and_prediction(response.output_text)
print(reasoning)
print(predictions)
