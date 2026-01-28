# LLM-Driven Adversarial Reinforcement Learning for IoT Intrusion Detection

This project works on reproducing and extending the research paper *"An adversarial environment reinforcement learning-driven intrusion detection algorithm for Internet of Things"*, adapting it to the **TON_IoT** dataset and integrating **LLM-based capabilities**.

## 🚀 Key Features
*   **Adversarial Reinforcement Learning**: DDQN Agent trained against an adversarial environment.
*   **TON_IoT Dataset**: Adapted to handle heterogeneous IoT telemetry and network data.
*   **LLM Extension (Multi-Layer)**:
    1.  **XAI (Explainability)**: Natural language explanations for IDS decisions.
    2.  **Reward Shaping**: Dynamic reward optimization to handle class imbalance.
    3.  **Red Team (Adversarial)**: Generative LLM that creates "hard" samples to robustify the agent.

## 🛠️ Installation

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Training (Synthetic Mode)
To test the pipeline without the full dataset:
```bash
python -m src.train --episodes 50 --synthetic
```

### 2. Training (Real Data)
If you have the `Train_Test_Network.csv` from TON_IoT:
```bash
python -m src.train --episodes 500 --data_path "path/to/Train_Test_Network.csv" --output_dir "./results"
```

## 📂 Project Structure
*   `src/`: Source code.
    *   `agents/`: DDQN Agent implementation (PyTorch).
    *   `envs/`: Gym Environment with Adversarial Logic.
    *   `llm/`: LLM modules (XAI, Reward, Adversary) and Mock interfaces.
*   `analysis.md`: Detailed Research Analysis, Architecture Diagrams, and Experimental Plan.
*   `implementation_plan.md`: Development roadmap.

## 🔬 Research Context
*   **Original Paper**: Chahira et al., 2024 (EURASIP Journal on Wireless Communications and Networking).
*   **Extensions**: Proposed LLM integration for Explainability and Robustness.

## ⚠️ Requirements
*   Python 3.8+
*   PyTorch
*   Gym
*   Pandas / Numpy / Scikit-Learn
