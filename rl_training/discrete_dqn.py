import pandas as pd
import tianshou as ts
import gymnasium as gym
from env import AlphaAblationEnv # Import your class
from tianshou.utils.net.common import Net
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy

def main():
    # 1. Load your actual data
    # Adjust path if running from inside rl_training/
    df = pd.read_csv("../data/factors.csv") 
    
    # 2. Setup Environment Factory
    def make_env():
        return AlphaAblationEnv(df, label_column='target') # Ensure 'target' matches your CSV

    train_envs = ts.env.DummyVectorEnv([make_env for _ in range(5)])
    test_envs = ts.env.DummyVectorEnv([make_env for _ in range(2)])

    # 3. Dynamic Space Info (Detects how many factors you have)
    dummy_env = make_env()
    state_shape = dummy_env.observation_space.shape
    action_shape = dummy_env.action_space.n

    # 4. Define Network
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128])
    optim = ts.algorithm.optim.AdamOptimizerFactory(lr=1e-3)
    
    policy = DiscreteQLearningPolicy(
        model=net, 
        action_space=dummy_env.action_space,
        eps_training=0.1, 
        eps_inference=0.05
    )
    
    algorithm = ts.algorithm.DQN(policy=policy, optim=optim)

    # 5. Collector & Trainer (Same as your example)
    train_collector = ts.data.Collector(
        algorithm, train_envs, ts.data.VectorReplayBuffer(20000, 5)
    )
    test_collector = ts.data.Collector(algorithm, test_envs)

    result = algorithm.run_training(
        ts.trainer.OffPolicyTrainerParams(
            training_collector=train_collector,
            test_collector=test_collector,
            max_epochs=10,
            epoch_num_steps=5000,
            batch_size=64,
        )
    )
    print(f"Finished: {result}")

if __name__ == "__main__":
    main()