import gym
from gym import spaces
import numpy as np


class RIS_MISO_PDA(gym.Env):
    def __init__(self, num_antennas,
                 num_RIS_elements,
                 num_users,
                 mismatch=False,
                 channel_est_error=False,
                 cascaded_channels=False,
                 beta_min=0.6,
                 theta_bar=0.0,
                 kappa_bar=1.5,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2,
                 seed=0):
        super(RIS_MISO_PDA, self).__init__()

        self._max_episode_steps = np.inf

        self.M = num_antennas
        self.L = num_RIS_elements
        self.K = num_users

        self.mismatch = mismatch
        self.channel_est_error = channel_est_error
        self.cascaded_channels = cascaded_channels
        self.beta_min = beta_min
        self.theta_bar = theta_bar
        self.kappa_bar = kappa_bar

        assert self.M == self.K

        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        power_size = 2 * self.K

        self.action_dim = 2 * self.M * self.K + 2 * self.L

        if self.cascaded_channels:
            channel_size = 2 * self.K * self.L * self.M
        else:
            channel_size = 2 * (self.L * self.M + self.L * self.K)

        self.state_dim = power_size + channel_size + self.action_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=float)
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.action_dim,), dtype=float)

        self.H_1 = None
        self.H_2 = None
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)
        self.Phi_mismatch = np.eye(self.L, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None

        self.info = {'episode': None, 'true reward': None}

        self.seed(seed)

    def seed(self, seed):
        np.random.seed(seed)

    def _compute_PDA(self, angles):
        betas = (1 - self.beta_min) * ((np.sin(angles - self.theta_bar) + 1) / 2) ** self.kappa_bar + self.beta_min

        return betas

    def _compute_D(self):
        D = np.diag(self.H_2[:, 0]) @ self.H_1

        for column_idx in np.arange(1, self.H_2.shape[1]):
            D = np.vstack((D, np.diag(self.H_2[:, column_idx] @ self.H_1)))

        if self.channel_est_error:
            D += np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)

        return D

    def _compute_H_2_tilde(self, D):
        if self.cascaded_channels:
            return np.diag(list(np.diag(self.Phi)) * self.K) @ D @ self.G
        else:
            return self.H_2.T @ self.Phi @ self.H_1 @ self.G

    def reset(self):
        self.episode_t = 0

        self.info["true reward"] = 0

        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.M))
        self.H_2 = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi))

        Phi_real = init_action[:, -2 * self.L:-self.L]
        Phi_imag = init_action[:, -self.L:]

        angles = np.arctan2(Phi_real, Phi_imag)
        betas = self._compute_PDA(angles)

        if self.mismatch:
            self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag) * betas
        else:
            self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)
            self.Phi_mismatch = self.Phi * betas

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        D = self._compute_D()
        H_2_tilde = self._compute_H_2_tilde(D)

        # power_r_real = np.real(H_2_tilde).reshape(1, -1) ** 2
        # power_r_imag = np.imag(H_2_tilde).reshape(1, -1) ** 2
        # power_r = np.hstack((power_r_real, power_r_imag))
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        if self.cascaded_channels:
            D_real, D_imag = np.real(D).reshape(1, -1), np.imag(D).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, D_real, D_imag))
        else:
            H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
            H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

            self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state

    def _compute_reward(self, Phi):
        reward = 0
        opt_reward = 0

        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)

            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2

            x = x.item()

            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            y = interference + (self.K - 1) * self.awgn_var

            rho_k = x / y

            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward

    def step(self, action, custom_betas=None):
        self.episode_t += 1

        action = action.reshape(1, -1)

        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)

        angles = np.arctan2(Phi_real, Phi_imag)
        betas = self._compute_PDA(angles)

        if self.mismatch:
            self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag) * betas
        else:
            self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

            if custom_betas is not None:
                Phi_real_custom = action[:, -2 * self.L:-self.L] / custom_betas
                Phi_imag_custom = action[:, -self.L:] / custom_betas

                actual_angles = np.arctan2(Phi_real_custom, Phi_imag_custom)
                actual_betas = self._compute_PDA(angles)

                self.Phi_mismatch = np.eye(self.L, dtype=complex) * (Phi_real_custom + 1j * Phi_imag_custom) * actual_betas
            else:
                self.Phi_mismatch = self.Phi * betas

        power_t =  np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        D = self._compute_D()
        H_2_tilde = self._compute_H_2_tilde(D)

        # power_r_real = np.real(H_2_tilde).reshape(1, -1) ** 2
        # power_r_imag = np.imag(H_2_tilde).reshape(1, -1) ** 2
        # power_r = np.hstack((power_r_real, power_r_imag))
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        if self.cascaded_channels:
            D_real, D_imag = np.real(D).reshape(1, -1), np.imag(D).reshape(1, -1)
            self.state = np.hstack((action, power_t, power_r, D_real, D_imag))
        else:
            H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
            H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

            self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        reward, opt_reward = self._compute_reward(self.Phi)

        if self.mismatch:
            true_reward = reward
        else:
            true_reward, opt_reward = self._compute_reward(self.Phi_mismatch)

        done = opt_reward == reward or self.episode_t >= self._max_episode_steps

        self.info["true reward"] = true_reward

        return self.state, reward, done, self.info

    def close(self):
        pass
