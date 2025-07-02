from pacman import PacmanEnv, STAY, UP, DOWN, LEFT, RIGHT
from rich import print

env = PacmanEnv()
print(env.reset())
while True:
    if env.is_terminated():
        print("Environment terminated.")
        break
    action = input("Select action: ")
    if action == "S":
        action = STAY
    elif action == "U":
        action = UP
    elif action == "D":
        action = DOWN
    elif action == "L":
        action = LEFT
    elif action == "R":
        action = RIGHT
    else:
        continue
    print(env.step({"player": action}))
    print(env.render())
