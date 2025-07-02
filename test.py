from pacman import PacmanEnv, STAY, UP, DOWN, LEFT, RIGHT
from rich import print

env = PacmanEnv()
print(env.reset())
while True:
    if env.is_terminated():
        print("Environment terminated.")
        break
    action = input("Select action: ")
    if action == "s":
        action = STAY
    elif action == "u":
        action = UP
    elif action == "d":
        action = DOWN
    elif action == "l":
        action = LEFT
    elif action == "r":
        action = RIGHT
    else:
        continue
    print(env.step({"player": action}))
    print(env.render())
