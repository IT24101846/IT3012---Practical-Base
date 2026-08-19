# simulator_search.py
# Step 1.3 Observation Task: compare BFS / DFS / UCS on the same map.

from visual_grid_game import VisualGridHuntGame
from agent import SearchAgent


def run(algo, width=10, height=10, num_food=8, num_traps=3, seed=None):
    import random
    if seed is not None:
        random.seed(seed)

    env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_traps=num_traps)
    agent = SearchAgent()
    agent.active_algo = algo

    print(f"\n=== SearchAgent [{algo}] ===")
    while not env.is_done():
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        print(f"step {env.steps:>2} | pos={env.agent_pos} | action={action:<6} "
              f"| food left={len(env.food_positions)} | score={env.score}")

    print(f"{algo} finished after {env.steps} steps -- "
          f"final score {env.score}, food left {len(env.food_positions)}")
    return env.score, env.steps


if __name__ == "__main__":
    # same seed => same map/food/traps for a fair side-by-side comparison
    for algo in ['BFS', 'DFS', 'UCS']:
        run(algo, seed=42)