import gym
from gym import spaces
import numpy as np
from src.data_loader import TONIoTLoader
from src.llm.adversary import RedTeamLMM

class AdversarialIDSEnv(gym.Env):
    def __init__(self, data_loader: TONIoTLoader, dataset_split='train', adversarial=True, red_team_llm=None):
        super(AdversarialIDSEnv, self).__init__()
        
        self.loader = data_loader
        self.split = dataset_split
        self.adversarial = adversarial
        self.red_team = red_team_llm
        self.reward_weights = {} # Default weights 1.0

        
        # Get data
        splits = self.loader.get_splits()
        if split == 'train':
            self.X, self.y = splits[0]
        elif split == 'val':
            self.X, self.y = splits[1]
        else:
            self.X, self.y = splits[2] # test
            
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_classes = len(np.unique(self.y))
        
        # Define Space
        self.action_space = spaces.Discrete(self.n_classes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_features,), dtype=np.float32)
        
        # Internal State
        self.current_idx = 0
        self.max_steps = 1000 # Episode length
        self.steps = 0
        
        # Adversarial Memory (Indices of hard samples)
        self.hard_samples = []

    def reset(self):
        self.steps = 0
        self.current_idx = np.random.randint(0, self.n_samples)
        return self.X[self.current_idx]

    def set_reward_weights(self, weights):
        """Updates class-specific reward weights."""
        self.reward_weights = weights

    def step(self, action):
        true_label = self.y[self.current_idx]
        
        # Reward Calculation with Shaping
        # Base reward
        if action == true_label:
            base_reward = 1.0
        else:
            base_reward = -1.0
            
        # Apply Weight if exists for this class (using index as key simplify)
        # Assuming weights are dict {class_idx: weight} or {class_name: weight}
        # Ideally we map class name, but let's use a multiplier.
        # For simplicity in this reproduction:
        weight = self.reward_weights.get(true_label, 1.0)
        reward = base_reward * weight

        if action != true_label and self.current_idx not in self.hard_samples:
            self.hard_samples.append(self.current_idx)
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        # Next State Service
        if self.adversarial:
            if self.red_team:
                # LLM Red Team Approach: Mutate the current hard sample or a new one
                # Logic: Pick a sample, mutate it via LLM
                 idx = np.random.choice(self.hard_samples) if len(self.hard_samples) > 0 else np.random.randint(0, self.n_samples)
                 original_features = self.X[idx]
                 # Only mutate occasionally to not destabilize completely
                 if np.random.rand() < 0.5:
                     next_state = self.red_team.generate_adversarial_sample(original_features)
                 else:
                     next_state = original_features
                 self.current_idx = idx # Keep track for next step logic (assumes independent samples though)
            elif len(self.hard_samples) > 0 and np.random.rand() < 0.3:
                self.current_idx = np.random.choice(self.hard_samples)
                next_state = self.X[self.current_idx]
            else:
                self.current_idx = np.random.randint(0, self.n_samples)
                next_state = self.X[self.current_idx]
        else:
            self.current_idx = np.random.randint(0, self.n_samples)
            next_state = self.X[self.current_idx]
        
        info = {
            'true_label': true_label,
            'is_success': action == true_label
        }
        
        return next_state, reward, done, info

    def render(self, mode='human'):
        pass
