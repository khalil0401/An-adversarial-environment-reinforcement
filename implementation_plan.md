# implementation_plan.md

# Goal Description
Reproduce the paper "An adversarial environment reinforcement learning-driven intrusion detection algorithm for Internet of Things" using the **TON_IoT** dataset instead of BoT-IoT, and extend it with a three-layer LLM architecture (XAI, Reward Shaping, Adversarial Self-Play).

## User Review Required
> [!IMPORTANT]
> **Dataset Adaptation**: The original paper uses BoT-IoT. We are switching to **TON_IoT**.
> *   **Impact**: Feature space and class labels will change.
> *   **Mitigation**: We will map TON_IoT features to a similar vector format and normalize them. The action space will change from BoT-IoT classes (DDoS, DoS, Recon, Theft) to TON_IoT classes (Backdoor, DDoS, DoS, Injection, MITM, Password, Ransomware, Scanning, XSS).

> [!WARNING]
> **Adversarial Logic**: The original paper uses a "static" adversarial environment (selecting hard samples).
> *   **Extension**: In Phase 3, we will replace this with a dynamic **LLM-driven Red Team** that *generates* new features, not just sampling.

## Phase 1: Paper Deconstruction (Methodology)

### 1. System Architecture
The system consists of two main agents interacting in a Reinforcement Learning framework:
*   **Agent (IDS)**: A Classifier (Policy Network) that predicts the label of a network flow.
*   **Environment (Adversarial)**: A mechanism that supplies network traffic samples. In the "Adversarial" mode, it selects samples that are historically hard to classify.

### 2. RL Formulation (MDP)
*   **State ($S$)**: A vector representing a single network traffic flow (feature vector from TON_IoT).
    *   $S_t = \{f_1, f_2, ..., f_n\}$ where $f_i$ are normalized features (e.g., duration, src_bytes, proto).
*   **Action ($A$)**: The predicted class label.
    *   $A_t \in \{0, 1, ..., K\}$ where $K$ is the number of attack classes in TON_IoT + Benign.
*   **Reward ($R$)**:
    *   Correct Prediction ($A_t == TrueLabel$): $R = +1$
    *   Incorrect Prediction ($A_t \neq TrueLabel$): $R = -1$
    *   *Adversarial Twist*: The Environment tries to minimize the Agent's cumulative reward by selecting difficult $S_{t+1}$.
*   **Policy ($\pi$)**: Deep Q-Network (or similar Deep RL policy) mapping $S \to A$.

## Phase 2: Project Reconstruction (TON_IoT)

### 1. Data Preprocessing (TON_IoT)
We will use the **TON_IoT Train_Test_Network** dataset (CSV format).
*   **Cleaning**: Handle missing values, infinite values.
*   **Encoding**: Label Encode categorical features (proto, service, conn_state).
*   **Normalization**: MinMax Scaling or Z-Score for numerical features.
*   **Splitting**: 70% Train, 15% Val, 15% Test.

### 2. Components
#### [NEW] `src/data_loader.py`
- Class `TONIoTLoader`: Handles loading CSV, preprocessing, and providing batches.

#### [NEW] `src/envs/ids_env.py` (Gym Interface)
- Class `AdversarialIDSEnv(gym.Env)`:
    - `reset()`: Resets stats, picks random initial sample.
    - `step(action)`:
        - Compares action to true label.
        - Computes Reward.
        - **Adversarial Step**: Selects next state $S_{t+1}$ based on difficulty (e.g., from a pool of misclassified samples).
        - Returns `(next_state, reward, done, info)`.

#### [NEW] `src/agents/dqn_agent.py`
- PyTorch implementation of DDQN (Double DQN).
- Replay Buffer.
- Epsilon-Greedy strategy.

#### [NEW] `src/train.py`
- Main training loop.
- Metric tracking (Accuracy, Precision, Recall, F1, Detection Rate).

## Phase 3: LLM Extension (The Contribution)

### Layer 1: Explainability (XAI)
*   **Module**: `src/llm/xai_explainer.py`
*   **Input**: State features (decoded), Predicted Action, True Label.
*   **Model**: Local LLM (Phi-3 or similar) or API.
*   **Output**: "The system classified this as DDoS because 'src_bytes' is abnormally high and 'proto' is UDP..."

### Layer 2: Reward Shaping
*   **Module**: `src/llm/reward_shaper.py`
*   **Logic**: Every $N$ episodes, analyze confusion matrix.
*   **Output**: Dynamic weights $W_c$ for each class $c$.
    *   $R = W_{true\_class} \times (\pm 1)$
    *   Helps with class imbalance (TON_IoT has rare attacks).

### Layer 3: Adversarial Self-Play (Red Team)
*   **Module**: `src/llm/red_team.py`
*   **Logic**: Instead of just sampling, the LLM takes a sample and *mutates* features to evade detection.
    *   "Change TCP window size to X to look like normal traffic."
*   **Impact**: Hardens the IDS against zero-day variations.

## Verification Plan

### Automated Tests
1.  **Environment Check**:
    ```bash
    pytest tests/test_env.py
    ```
    *   Verify `env.step()` returns valid shape (state_dim).
    *   Verify reward logic (+1/-1).

2.  **Training Pipeline**:
    ```bash
    python src/train.py --test_mode
    ```
    *   Run 10 episodes. Ensure loss decreases or fluctuates (not NaN).

3.  **LLM Connectivity**:
    ```bash
    pytest tests/test_llm.py
    ```
    *   Verify dummy prompt returns string response.

### Manual Verification
*   **Data Inspection**: Manually check `data_loader` output histogram to ensure classes are mapped correctly.
*   **XAI Output**: Read `logs/explanations.txt` to verify the explanations make sense linguistically.
