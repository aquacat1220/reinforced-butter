from pacman import PacmanEnv, STAY, UP, DOWN, LEFT, RIGHT, PursueGhost, PatrolPowerGhost
from rich import print

env = PacmanEnv()

observations, _ = env.reset()
dones: dict[str, bool] = {ghost: False for ghost in env.ghosts}
dones[env.player] = False

ghosts = {ghost_name: PatrolPowerGhost() for ghost_name in env.ghosts}

print(observations)
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
    actions: dict[str, int] = {
        ghost_name: ghost.get_action(observations[ghost_name])
        for (ghost_name, ghost) in ghosts.items()
        if not dones[ghost_name]
    }
    actions[env.player] = action
    observations, _, terminateds, truncateds, _ = env.step(actions)
    dones = {
        agent: terminateds[agent] or truncateds[agent] or dones[agent]
        for agent in terminateds
    }
    print(observations)
    print(env.render())
