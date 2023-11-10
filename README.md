# Joint Transmit Beamforming and Phase Shifts Design with Deep Reinforcement Learning Under the Phase-Dependent Amplitude Model

### Proceedings are out!
If you use our code/results, please cite the paper.
```
@INPROCEEDINGS{10283517,
  author={Saglam, Baturay and Gurgunoglu, Doga and Kozat, Suleyman S.},
  booktitle={2023 IEEE International Conference on Communications Workshops (ICC Workshops)}, 
  title={Deep Reinforcement Learning Based Joint Downlink Beamforming and RIS Configuration in RIS-Aided MU-MISO Systems Under Hardware Impairments and Imperfect CSI}, 
  year={2023},
  volume={},
  number={},
  pages={66-72},
  doi={10.1109/ICCWorkshops57953.2023.10283517}
}
```

##
PyTorch implementation of the paper, [_Deep Reinforcement Learning Based Joint Downlink Beamforming and RIS Configuration in RIS-aided MU-MISO Systems Under Hardware Impairments and Imperfect CSI_](https://ieeexplore.ieee.org/document/10283517). The paper has been accepted to 2023 IEEE International Conference on Communications the 5th Workshop on Data Driven Intelligence for Networks and Systems (DDINS).

For the first time in the literature, we solve a Reconfigurable Intelligent Surface (RIS) assisted multi-user multi-input single-output (MISO) System problem under _hardware impairments_ through a machine learning approach. Specifically, the deep reinforcement learning algorithm of [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) combined with [DISCOVER](https://link.springer.com/article/10.1007/s10994-023-06363-4) used to tackle the issues induced by the [phase-dependent amplitude model (PDA)](https://ieeexplore.ieee.org/document/9148961) in RIS-aided systems. 

The algorithm is tested, and the results are produced on a custom RIS-assisted multi-user MISO environment. 
Learning curves for the results presented in the paper are found under [./Learning Curves](https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning/tree/main/Learning%20Curves). Each learning curve is formatted as NumPy arrays of 20000 instant rewards (20000,).
Corresponding learning figures are found under [./Learning Figures](https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning/tree/main/Learning%20Figures). The learning curves depict instant rewards achieved by the agents for 20000 training steps, averaged over ten random seeds.

### Pseudocode
<img src="https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning/blob/main/pseudocode.jpg" width="450">

### The Hyper-Parameter Setting
| Hyper-Parameter  | Value |
| ------------- | ------------- |
| # of hidden layers (all networks) | $2$ |
| # of units in each hidden layer (all networks) | $256$ |
| Hidden layers activation (all networks) | ReLU | 
| Final layer activation (Q-networks) | Linear |
| Final layer activation (actor, explorer) | tanh |
| Learning rate $\eta$ (all networks) | $10^{-3}$ |
| Weight decay (all networks) | None |
| Weight initialization (all networks) | Xavier uniform |
| Bias initialization (all networks) | constant |
| Optimizer (all networks) | Adam |
| Total time steps per training | $20000$ |
| Experience replay buffer size | $20000$ |
| Experience replay sampling method | uniform |
| Mini-batch size | $16$ |
| Discount term $\gamma$ | $1$ |
| Learning rate for target networks $\tau$ (all networks) | $10^{-3}$ |
| Network update interval (all networks) | after each environment step |
| Initial $\alpha$ | $0.2$ |
| Entropy target | $\texttt{-action dimension}$ |
| SAC log standard deviation clipping | $(-20, 2)$ |
| SAC $\epsilon$ | $10^{-6}$ |
| $\beta$-Space Exploration $\lambda$ at time step $t$ | $0.3 - \frac{0.3 \times t}{\texttt{total time steps}} $ |
| $\mu$ (environment-related) | 0 |
| $\kappa$ (environment-related) | 1.5 |
| Channel noise variance $\sigma_{e}^{2}$ (environment-related) | $10^{-2}$ |
| AWGN channel variance $\sigma_{w}^{2}$ (environment-related) | $10^{-2}$ |
| Channel matrix initialization (Rayleigh) (environment-related) | $\mathcal{CN}(0, 1)$ |



### Computing Infrastructure
The hardware/software model/version alters the DRL agents' training stochasticity due to the use of random seeds. Therefore, it complicates the precise reproduction of the reported results. The following computing infrastructure is used to produce the results.

| Hardware/Software  | Model/Version |
| ------------- | ------------- |
| Operating System  | Ubuntu 18.04.5 LTS  |
| CPU  | AMD Ryzen 7 3700X 8-Core Processor |
| GPU  | Nvidia GeForce RTX 2070 SUPER |
| CUDA  | 11.1  |
| Python  | 3.8.5 |
| PyTorch  | 1.8.1 |
| OpenAI Gym  | 0.17.3 |
| MuJoCo  | 1.50 |
| Box2D  | 2.3.10 |
| NumPy  | 1.19.4 |


### Run
**0. Requirements**
  ```bash
gym==0.17.3
numpy==1.23.3
torch==1.12.1
  ```
  
**1. Installing** 
* Clone this repo: 
    ```bash
    git clone https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning
    cd RIS-MISO-PDA-Deep-Reinforcement-Learning
    ```
* Install Python requirements: 
    ```bash
    pip install -r requirements.txt
    ```

**2. Register the custom RIS-assisted multi-user MISO environment to OpenAI Gym**

You need to use the [environment.py](https://github.com/baturaysaglam/RIS-MISO-PDA-Deep-Reinforcement-Learning/blob/main/environment.py) file to register the environment to OpenAI Gym. A tutorial on how to register an environment can be found [here](https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym). 
    
**3. Train the model from scratch**
  * Usage:
   ```
usage: main.py [-h] [--objective_function OBJECTIVE_FUNCTION]
               [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--max_time_steps N] [--buffer_size BUFFER_SIZE]
               [--batch_size N] [--save_model SAVE_MODEL]
               [--load_model LOAD_MODEL] [--num_antennas N]
               [--num_RIS_elements N] [--num_users N] [--power_t N]
               [--awgn_var G] [--channel_noise_var G] [--mismatch N]
               [--channel_est_error N] [--cascaded_channels N] [--beta_min N]
               [--theta_bar N] [--kappa_bar N] [--discount G] [--tau G]
               [--actor_lr G] [--critic_lr G] [--decay G]
               [--policy_type POLICY_TYPE] [--target_update_interval N]
               [--alpha G] [--automatic_entropy_tuning G]
               [--exp_regularization_term G]
               [--linear_schedule_exp_regularization G]
  ```
  * Optional arguments:
  ```
optional arguments:
  -h, --help            show this help message and exit
  --objective_function OBJECTIVE_FUNCTION
                        Is PDA assumed?
  --policy POLICY       Algorithm (default: Beta-Space Exploration)
  --env ENV             Environment name
  --seed SEED           Seed number for PyTorch and NumPy (default: 0)
  --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
  --max_time_steps N    Number of training steps (default: 20000)
  --buffer_size BUFFER_SIZE
                        Size of the experience replay buffer (default: 20000)
  --batch_size N        Mini-batch size (default: 16)
  --save_model SAVE_MODEL
                        Save model and optimizer parameters
  --load_model LOAD_MODEL
                        Model load file name; if empty, does not load
  --num_antennas N      Number of antennas in the BS (default: 4)
  --num_RIS_elements N  Number of RIS elements (default: 4)
  --num_users N         Number of users (default: 4)
  --power_t N           Transmission power for the constrained optimization
                        (in dBm, default: 30)
  --awgn_var G          Variance of the additive white Gaussian noise
                        (default: 0.01)
  --channel_noise_var G
                        Variance of the noise in the cascaded channels
                        (default: 0.01)
  --mismatch N          Is PDA assumed?
  --channel_est_error N
                        Is channel estimation error assumed?
  --cascaded_channels N
                        Is cascaded channels assumed?
  --beta_min N          Minimum beta value in the PDA calculation (default:
                        0.6)
  --theta_bar N         Theta bar value in the PDA calculation (default: 0.0)
  --kappa_bar N         Kappa bar value in the PDA calculation (default: 1.5)
  --discount G          Discount factor for reward (default: 1.0)
  --tau G               Learning rate in soft/hard updates of the target
                        networks (default: 0.001)
  --actor_lr G          Learning rate for the actor (and explorer) network
                        (default: 0.001)
  --critic_lr G         Learning rate for the critic network (default: 0.001)
  --decay G             Decay rate for the networks (default: 0.0)
  --policy_type POLICY_TYPE
                        SAC Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --target_update_interval N
                        SAC Number of critic function updates per training
                        time step (default: 1)
  --alpha G             SAC Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        SAC Automatically adjust α (default: False)
  --exp_regularization_term G
                        Initial value for the exploration regularization term
                        (default: 0.3)
  --linear_schedule_exp_regularization G
                        Linearly schedule exploration regularization term
```
