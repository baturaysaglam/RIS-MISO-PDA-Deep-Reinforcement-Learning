import argparse
import os

import gym
import numpy as np
import torch

import SAC
import Beta_Space_Exp_SAC
import utils


def whiten(state):
    return (state - np.mean(state)) / np.std(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--objective_function", default="mismatch", type=str, help='Is PDA assumed?')

    # Training-specific parameters
    parser.add_argument("--policy", default="Beta_Space_Exp_SAC", help='Algorithm (default: Beta-Space Exploration)')
    parser.add_argument("--env", default="RIS_MISO_PDA-v0", help='Environment name')
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--max_time_steps", default=int(2e4), type=int, metavar='N', help='Number of training steps (default: 20000)')
    parser.add_argument("--buffer_size", default=int(2e4), type=int, help='Size of the experience replay buffer (default: 20000)')
    parser.add_argument("--batch_size", default=16, metavar='N', help='Mini-batch size (default: 16)')
    parser.add_argument("--save_model", default=False, type=bool, help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    # Environment-specific parameters
    parser.add_argument("--num_antennas", default=4, type=int, metavar='N', help='Number of antennas in the BS (default: 4)')
    parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='Number of RIS elements (default: 4)')
    parser.add_argument("--num_users", default=4, type=int, metavar='N', help='Number of users (default: 4)')
    parser.add_argument("--power_t", default=30, type=float, metavar='N', help='Transmission power for the constrained optimization (in dBm, default: 30)')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_noise_var", default=1e-2, type=float, metavar='G', help='Variance of the noise in the cascaded channels (default: 0.01)')

    # Phase-Dependent RIS Amplitude Model parameters
    parser.add_argument("--mismatch", default=True, type=bool, metavar='N', help='Is PDA assumed?')
    parser.add_argument("--channel_est_error", default=True, type=bool, metavar='N', help='Is channel estimation error assumed?')
    parser.add_argument("--cascaded_channels", default=True, type=bool, metavar='N', help='Is cascaded channels assumed?')
    parser.add_argument("--beta_min", default=0.6, type=float, metavar='N', help='Minimum beta value in the PDA calculation (default: 0.6)')
    parser.add_argument("--theta_bar", default=0.0, type=float, metavar='N', help='Theta bar value in the PDA calculation (default: 0.0)')
    parser.add_argument("--kappa_bar", default=1.5, type=float, metavar='N', help='Kappa bar value in the PDA calculation (default: 1.5)')

    # Algorithm-specific parameters
    parser.add_argument("--discount", default=1.0, metavar='G', help='Discount factor for reward (default: 1.0)')
    parser.add_argument("--tau", default=1e-3, type=float, metavar='G', help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--actor_lr", default=1e-3, type=float, metavar='G',help='Learning rate for the actor (and explorer) network (default: 0.001)')
    parser.add_argument("--critic_lr", default=1e-3, type=float, metavar='G', help='Learning rate for the critic network (default: 0.001)')
    parser.add_argument("--decay", default=0.0, type=float, metavar='G', help='Decay rate for the networks (default: 0.0)')

    # SAC-specific parameters
    parser.add_argument('--policy_type', default="Gaussian", help='SAC Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='SAC Number of critic function updates per training time step (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='SAC Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G', help='SAC Automatically adjust α (default: False)')

    # Beta-Space Exploration-specific parameters
    parser.add_argument("--exp_regularization_term", default=0.3, type=float, metavar='G', help='Initial value for the exploration regularization term (default: 0.3)')
    parser.add_argument("--linear_schedule_exp_regularization", default=True, type=bool, metavar='G', help='Linearly schedule exploration regularization term')

    args = parser.parse_args()

    if args.objective_function == "mismatch":
        args.mismatch = False
    elif args.objective_function == "golden":
        args.channel_est_error = False

    if "Beta_Space" in args.policy:
        args.mismatch = False
        args.channel_est_error = True

    print("-----------------------------------------------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Scenario: {args.objective_function.capitalize()}")
    print("-----------------------------------------------------------------------------")

    file_name = f"{args.policy}_{args.objective_function}_{args.seed}"

    save_path = f"Beta_min. = {args.beta_min}, K = {args.num_users}, M = {args.num_antennas}, N = {args.num_RIS_elements}, P_t = {float(args.power_t)}"

    if args.save_model and not os.path.exists(f"./Models/{save_path}"):
        os.makedirs(f"./Models/{save_path}")

    if not os.path.exists(f"./Results/{save_path}"):
        os.makedirs(f"./Results/{save_path}")

    environment_kwargs = {
        "num_antennas": args.num_antennas,
        "num_RIS_elements": args.num_RIS_elements,
        "num_users": args.num_users,
        "mismatch": args.mismatch,
        "channel_est_error": args.channel_est_error,
        "cascaded_channels": args.cascaded_channels,
        "beta_min": args.beta_min,
        "theta_bar": args.theta_bar,
        "kappa_bar": args.kappa_bar,
        "AWGN_var": args.awgn_var,
        "channel_noise_var": args.channel_noise_var,
        "seed": args.seed,
    }

    env = gym.make(args.env, **environment_kwargs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    agent_kwargs = {
        "state_dim": state_dim,
        "action_space": env.action_space,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "power_t": args.power_t,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "policy_type": args.policy_type,
        "alpha": args.alpha,
        "target_update_interval": args.target_update_interval,
        "automatic_entropy_tuning": args.automatic_entropy_tuning,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }

    # Initialize the algorithm
    if args.policy == "SAC":
        agent = SAC.SAC(**agent_kwargs)
        replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    elif args.policy == "Beta_Space_Exp_SAC":
        agent = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(**agent_kwargs, beta_min=args.beta_min)
        replay_buffer = utils.BetaExperienceReplayBuffer(state_dim, action_dim, args.num_RIS_elements, args.buffer_size)
    else:
        raise NotImplementedError("invalid algorithm name")

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    instant_rewards = []
    instant_mismatch_rewards = []

    state, done = env.reset(), False

    max_reward = 0
    max_mismatch_reward = 0

    episode_time_steps = 0
    episode_num = 0

    state = whiten(state)

    exp_regularization = args.exp_regularization_term

    for t in range(int(args.max_time_steps)):
        episode_time_steps += 1

        if args.policy == "SAC":
            action = agent.select_action(state)
        elif "Beta_Space" in args.policy:
            action, beta = agent.select_action(state, exp_regularization)
        else:
            action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * args.exploration_noise, size=action_dim)).clip(-max_action, max_action)

        # Take the selected action
        if "Beta_Space" in args.policy:
            next_state, reward, done, info = env.step(action, beta)
        else:
            next_state, reward, done, info = env.step(action)

        mismatch_reward = info["true reward"]

        next_state = whiten(next_state)

        instant_rewards.append(reward)
        instant_mismatch_rewards.append(mismatch_reward)

        if reward > max_reward:
            max_reward = reward

        if mismatch_reward > max_mismatch_reward:
            max_mismatch_reward = mismatch_reward

        reward = reward - np.mean(instant_rewards)

        # Store data in the experience replay buffer
        if "Beta_Space" in args.policy:
            replay_buffer.add(state, action, beta, next_state, reward, float(done))
        else:
            replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state

        if (t + 1) % 100 == 0:
            print(f"Time step: {t + 1} Max. Reward: {max_reward:.3f} Max. Mismatch Reward: {max_mismatch_reward:.3f}")

            np.save(f"./Results/{save_path}/{file_name}", instant_mismatch_rewards)

            if args.save_model:
                agent.save(f"./Models/{save_path}/{file_name}")

        # Train the agent after collecting sufficient samples
        if "Beta_Space" in args.policy:
            agent.update_parameters(replay_buffer, exp_regularization, args.batch_size)
        else:
            agent.update_parameters(replay_buffer, args.batch_size)

        if args.linear_schedule_exp_regularization:
            exp_regularization = args.exp_regularization_term - (args.exp_regularization_term * (t / args.max_time_steps))
