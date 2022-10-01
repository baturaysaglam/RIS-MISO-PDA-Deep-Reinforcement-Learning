import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from utils import soft_update, hard_update, weights_init_

# Implementation of the Soft Actor-Critic algorithm (SAC)
# Paper: https://arxiv.org/abs/1801.01290


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, M, N, K, power_t, device, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.device = device

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def compute_power(self, action):
        # Normalize the power
        G_real = action[:, :self.M ** 2].cpu().data.numpy()
        G_imag = action[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()

        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)

        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))

        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)

        return current_power_t

    def _compute_power(self, action):
        # Normalize the power
        G_real = action[:, :self.M ** 2].cpu().data.numpy()
        G_imag = action[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()

        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)

        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))

        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)

        return current_power_t

    def compute_phase(self, action):
        # Normalize the phase matrix
        Phi_real = action[:, -2 * self.N:-self.N].detach()
        Phi_imag = action[:, -self.N:].detach()

        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)

    def _compute_phase(self, action):
        # Normalize the phase matrix
        Phi_real = action[:, -2 * self.N:-self.N].detach()
        Phi_imag = action[:, -self.N:].detach()

        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        # Normalize the transmission power and phase matrix
        current_power_t = self.compute_power(action.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)

        real_normal, imag_normal = self.compute_phase(action.detach())

        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)

        division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)

        action /= division_term

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class SAC(object):
    def __init__(self, state_dim,
                 action_space,
                 M,
                 N,
                 K,
                 power_t,
                 actor_lr,
                 critic_lr,
                 policy_type,
                 alpha,
                 target_update_interval,
                 automatic_entropy_tuning,
                 device,
                 discount=0.99,
                 tau=0.001):

        power_t = 10 ** (power_t / 10)

        hidden_size = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        # Initialize the training parameters
        self.gamma = discount
        self.tau = tau
        self.alpha = alpha

        # Initialize the policy-specific parameters
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Set CUDA device
        self.device = device

        self.updates = 0

        # Initialize critic networks and optimizer
        self.critic = Critic(state_dim, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = Critic(state_dim, action_space.shape[0], hidden_size).to(self.device)

        hard_update(self.critic_target, self.critic)

        # Initialize actor network and optimizer
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=actor_lr)

        self.actor = GaussianPolicy(state_dim, action_space.shape[0], hidden_size, M, N, K, power_t, self.device).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

    def select_action(self, state, evaluate=False):
        self.actor.eval()

        state = torch.FloatTensor(state).to(self.device)

        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size=16):
        self.actor.train()

        # Sample from the experience replay buffer
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            # Select the target smoothing regularized action according to policy
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)

            # Compute the target Q-value
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Get the current Q-value estimates
        qf1, qf2 = self.critic(state_batch, action_batch)

        # Compute the critic loss
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Compute the critic loss
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Compute policy loss
        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Tune the temperature coefficient
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        self.updates += 1

        # Soft update the target critic network
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    # Save the model parameters
    def save(self, file_name):
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

    # Load the model parameters
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = hard_update.deepcopy(self.critic)
