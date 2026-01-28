import argparse
import os
import numpy as np
import torch
from src.data_loader import TONIoTLoader
from src.envs.ids_env import AdversarialIDSEnv
from src.agents.dqn_agent import DQNAgent
from src.llm.xai import XAIExplainer
from src.llm.reward import RewardShaper
from src.llm.adversary import RedTeamLMM
from src.llm.llm_interface import MockLLM, LocalLLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(agent, env, n_episodes=5):
    """Evaluates the agent on the current environment."""
    # env.split should ideally be 'test' but for simplicity here we assume env is set up
    total_rewards = []
    all_preds = []
    all_labels = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state) # Epsilon should be 0 or low effectively if we force it, but assume agent handles it
            
            # Temporarily disable epsilon for eval if needed, but agent.select_action uses self.epsilon
            # Ideally store old epsilon, set to 0, then restore.
            old_eps = agent.epsilon
            agent.epsilon = 0.0
            action = agent.select_action(state)
            agent.epsilon = old_eps
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            all_preds.append(action)
            all_labels.append(info['true_label'])
            
        total_rewards.append(episode_reward)
        
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return np.mean(total_rewards), accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Train Adversarial IDS")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--data_path", type=str, default=None, help="Path to TON_IoT CSV")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--use_llm", action="store_true", help="Use real Local LLM (requires GPU & transformers)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Init LLMs
    if args.use_llm:
        print("Initializing REAL Local LLM (Phi-3)... This may take time.")
        llm_engine = LocalLLM() # Uses transformers
    else:
        print("Using MOCK LLM (Fast, no GPU required). Use --use_llm to enable real model.")
        llm_engine = MockLLM()

    xai = XAIExplainer(llm_engine)
    reward_shaper = RewardShaper(llm_engine)
    red_team = RedTeamLMM(llm_engine)
    
    # 1. Load Data
    loader = TONIoTLoader(data_path=args.data_path, synthetic=args.synthetic or (args.data_path is None))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.get_splits()
    print(f"Data Loaded. Classes: {len(np.unique(y_train))}")
    
    # 2. Init Env and Agent
    # Pass Red Team to Env
    env = AdversarialIDSEnv(loader, dataset_split='train', adversarial=True, red_team_llm=red_team)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DQNAgent(input_dim, output_dim)
    
    print(f"Starting Training for {args.episodes} episodes...")
    
    # 3. Training Loop
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        loss = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss += agent.train()
            
            state = next_state
            total_reward += reward
        
        # End of Episode: Reward Shaping & XAI sample
        reward_shaper.update_history({'reward': total_reward, 'loss': loss})
        
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()
            
            # XAI: Explain the last decision of the episode
            explanation = xai.explain_prediction(state, action, info['true_label'])
            print(f"   [XAI] {explanation}")
            
            # Reward Shaping: Update weights
            new_weights = reward_shaper.compute_weights()
            if new_weights:
                env.set_reward_weights(new_weights)
                print(f"   [REWARD] Optimized weights: {new_weights}")
            
        # Log Progress
        print(f"Episode {episode+1}/{args.episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f} - Loss: {loss:.4f}")
        
        # Periodic Eval
        if (episode + 1) % 10 == 0:
            val_reward, acc, prec, rec, f1 = evaluate(agent, env, n_episodes=5)
            print(f"   [EVAL] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            
            # Save Checkpoint
            torch.save(agent.q_network.state_dict(), os.path.join(args.output_dir, f"checkpoint_ep{episode+1}.pth"))
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
