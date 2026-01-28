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

### Option A: Local (CPU/No-GPU)
Suitable for development and testing logic.
1.  **Clone** the repository.
2.  **Install Base Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Option B: Local (GPU + Real LLM)
Requires CUDA-capable GPU (VRAM >= 8GB recommended for Phi-3).
1.  **Install Base Dependencies + Transformers**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` includes `transformers` and `accelerate`)*

---

## 🏃 Usage

### 1. Synthetic Test (Quick Verify)
Runs training with synthetic data. Pass `--use_llm` to enable the real Phi-3 model (requires GPU).
```bash
# Mock LLM (Fast, CPU)
python -m src.train --episodes 10 --synthetic

# Real LLM (Slow, GPU Required)
python -m src.train --episodes 10 --synthetic --use_llm
```

### 2. Full Training (Real Dataset)
Download the `Train_Test_Network.csv` from the [TON_IoT Dataset](https://research.unsw.edu.au/projects/toniot-datasets).

```bash
# Mock LLM
python -m src.train --episodes 500 --data_path "path/to/Train_Test_Network.csv"

# Real LLM (Recommended for Research Results)
python -m src.train --episodes 500 --data_path "path/to/Train_Test_Network.csv" --use_llm
```

---

## ☁️ Kaggle Cloud Deployment (Recommended)
This project includes a pre-configured notebook for running experiments on Kaggle's free GPUs (P100/T4).

1.  **Get the Notebook**: Locate `kaggle_training.ipynb` in the repo root.
2.  **Upload to Kaggle**: Create a new Notebook on Kaggle and upload this file.
3.  **Add Dataset**: Search and add the "TON_IoT Network Dataset" to your Kaggle environment.
4.  **Enable GPU**: Settings -> Accelerator -> GPU P100 or T4.
5.  **Run**: Execute the notebook cells. It handles cloning and dependency installation automatically.

---

## 📂 Project Structure
*   `src/`: Source code.
    *   `agents/`: DDQN Agent implementation (PyTorch).
    *   `envs/`: Gym Environment with Adversarial Logic.
    *   `llm/`: LLM modules (XAI, Reward, Adversary) and Native/Mock interfaces.
*   `kaggle_training.ipynb`: **[NEW]** Deployment notebook for cloud training.
*   `analysis.md`: Detailed Research Analysis, Architecture Diagrams, and Experimental Plan.
*   `implementation_plan.md`: Development roadmap.

## 🔬 Research Context
*   **Original Paper**: Chahira et al., 2024 (EURASIP Journal on Wireless Communications and Networking).
*   **Extensions**: 
    *   **Generative Adversarial Red Team**: Uses Phi-3 to mutate attack signatures dynamically.
    *   **Explainable AI (XAI)**: Generates text justifications for detection alerts.
    *   **Reward Shaping**: Adapts RL incentives based on class imbalance analysis.

## ⚠️ Requirements
*   Python 3.8+
*   PyTorch 2.0+
*   Gymnasium / Gym
*   Transformers & Accelerate (for Local LLM)
*   Pandas / Numpy / Scikit-Learn
