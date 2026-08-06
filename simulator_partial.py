# simulator_partial.py

from visual_grid_game import VisualGridHuntGame
from agent import SimpleReflexAgent, ModelBasedAgent


def run(agent_class, label, width=10, height=10, num_food=8, num_traps=3):
    env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_traps=num_traps)
    agent = agent_class()

    print(f"\n=== {label} ===")
    recent_actions = []
    loop_flagged = False

    while not env.is_done():
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        recent_actions.append(action)

        print(f"step {env.steps:>2} | percept={percept} | action={action:<6} "
              f"| pos={env.agent_pos} | score={env.score}")

        if len(recent_actions) >= 6 and len(set(recent_actions[-6:])) == 1 and not loop_flagged:
            print("   -> same action six times in a row: STUCK IN A LOOP")
            loop_flagged = True

    print(f"{label} finished after {env.steps} steps -- "
          f"final score {env.score}, food left {len(env.food_positions)}")


if __name__ == "__main__":
    run(SimpleReflexAgent, "SimpleReflexAgent (Step 1.2)")
    run(ModelBasedAgent, "ModelBasedAgent (Step 1.3)")