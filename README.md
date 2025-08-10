# REMEDI-entero

**Entero-augmented** REinforcement learning-driven adaptive MEtabolism modeling of primary sclerosing cholangitis DIsease progression

-----

## Description 📝

Primary sclerosing cholangitis (PSC) is a rare, incurable disease wherein altered bile acid metabolism contributes to sustained liver injury. REMEDI captures bile acid dynamics and the body's adaptive response during PSC progression. REMEDI merges a differential equation (DE)-based mechanistic model that describes bile acid metabolism with reinforcement learning (RL) to continuously emulate the body's adaptations to PSC. An objective of adaptation is to maintain homeostasis by regulating enzymes involved in bile acid metabolism. These enzymes correspond to parameters of the DEs. REMEDI leverages RL to approximate adaptations in PSC, treating homeostasis as a reward signal and the adjustment of the DE parameters as the corresponding actions.

Compared to the original REMEDI model, the entero-augmented REMEDI model supports **PSC-specific gut dysbiosis parameters** and also features a **higher-resolution intestinal reabsorption model** with active and passive uptake refinements.

-----

## Installation on AWS ☁️

**AMI**: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04) 20250427`

1.  Connect to your AWS instance.

    ```bash
    # Navigate to where your private key is stored
    cd Library/Mobile\ Documents/com\~apple\~CloudDocs/Documents/2025H1/ 

    # Connect via SSH, forwarding the port for Jupyter
    ssh -i "aws_c6i.4xlarge.pem" -L 8899:localhost:8888 ubuntu@<AWS cluster>.amazonaws.com
    ```

2.  Start a new `tmux` session and launch Jupyter Lab.

    ```bash
    tmux new -s jupyter
    source /opt/pytorch/bin/activate
    jupyter-lab --no-browser --port=8888 
    ```

    *Access Jupyter Lab in your local browser at `localhost:8899` and set a password when prompted.*

3.  Clone the repository and install the required packages.

    ```bash
    git clone https://github.com/chang-hu/REMEDI-entero.git

    pip install --upgrade pip wheel setuptools
    pip install --no-cache-dir "numpy==2.1.0"
    pip install --upgrade "stable-baselines3[extra]>=2.3.0"
    pip install wandb
    ```

-----

## Workflow ⚙️

1.  Connect to your instance and start Jupyter Lab.

    ```bash
    ssh -i "aws_c6i.4xlarge.pem" -L 8888:localhost:8888 ubuntu@<AWS cluster>.compute-1.amazonaws.com
    tmux new -s jupyter
    jupyter-lab --no-browser --port=8888
    ```

2.  Replace `WANDB_API_KEY` in `REMEDI-entero/REMEDI_model.py` with your personal Weights & Biases API key.

3.  Run the execution script `REMEDI-entero/sb3_exe.sh` using the parameters defined in `REMEDI-entero/sb3.sh`.

      * First, make the script executable:
        ```bash
        chmod +x sb3_exe.sh
        ```
      * Then, run the script:
        ```bash
        ./sb3_exe.sh
        ```
      * **To run with microbiome changes**, set the following variables:
        ```bash
        GUT_DECONJ_FREQ_CO_MULTIPLIER=0.835
        GUT_BIOTR_FREQ_CA_MULTIPLIER=0.2
        ```
      * **To run without microbiome changes**, use the default values:
        ```bash
        GUT_DECONJ_FREQ_CO_MULTIPLIER=1.0
        GUT_BIOTR_FREQ_CA_MULTIPLIER=1.0
        ```
4. Use the notebook `REMEDI_analysis.ipynb` to analyze the trained RL model.

-----

## Alternative Usage ▶️

  * Use the script `REMEDI_model.py` to train an RL model with specified parameters.
  * Use the notebook `REMEDI_analysis.ipynb` to analyze the trained RL model.

The following command-line arguments can be used to customize the training process with `REMEDI_model.py`:

```sh
mkdir experiments
python REMEDI_model.py [--algorithm ALGORITHM] [--train_steps TRAIN_STEPS] [--learning_rate LEARNING_RATE] [--n_envs N_ENVS] [--adaptation_days ADAPTATION_DAYS] [--data_ID DATA_ID] [--max_ba_flow MAX_BA_FLOW] [--continue_train_suffix CONTINUE_TRAIN_SUFFIX] [--gut_deconj_freq_co_multiplier GUT_DECONJ_FREQ_CO_MULTIPLIER] [--gut_biotr_freq_CA_multiplier GUT_BIOTR_FREQ_CA_MULTIPLIER]
```

### Argument Descriptions

1.  `--algorithm ALGORITHM`: Specifies the RL algorithm for training the model (Default: `PPO`).
2.  `--train_steps TRAIN_STEPS`: Sets the number of training steps (Default: `2000000`).
3.  `--learning_rate LEARNING_RATE`: Sets the learning rate for the model (Default: `0.002`).
4.  `--n_envs N_ENVS`: Specifies the number of vectorized training environments to be used (Default: `15`).
5.  `--adaptation_days ADAPTATION_DAYS`: Sets the number of adaptation days (Default: `240`).
6.  `--data_ID DATA_ID`: Specifies the patient identifier (Default: `median`).
7.  `--max_ba_flow MAX_BA_FLOW`: Sets the maximum amount of bile acids allowed to pass through the bile duct to the ileum, in proportion to the degree of bile duct obstruction (Default: `10.0`).
8.  `--continue_train_suffix CONTINUE_TRAIN_SUFFIX`: Specifies the checkpoint identifier to resume training from a saved model; skip this option to start a new training session (Default: *None*).
9. `--gut_deconj_freq_co_multiplier`: Scales the colon bile-salt-hydrolase (BSH)–driven deconjugation rate (Default: 0.835)
10. `--gut_biotr_freq_CA_multiplier`: Scales the primary-to-secondary bile acid conversion rate (Default: 0.2)

*Default settings reproduce the results presented in REMEDI.*

-----

## Files Overview 📂

1.  **REMEDI\_model.py**: Main script to train an RL agent to emulate the body's adaptations.
2.  **REMEDI\_analysis.ipynb**: Notebook to visualize and analyze the trained RL model.
3.  **src/sb3\_BA\_ode.py**: Defines the bile acid metabolism system extended with PSC pathophysiology using a system of differential equations (DEs).
4.  **src/rl\_env.py**: Creates the RL environment with the bile acid DEs and specifies the step method, initialization, state/action space, reward calculation, etc.
5.  **src/rl\_util.py**: Contains functions for loading bile acid data and helper functions for logging.
6.  **src/rl\_eval.py**: Includes functions for visualizing RL results.