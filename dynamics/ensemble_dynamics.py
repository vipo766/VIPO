import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from typing import Callable, List, Tuple, Dict, Optional
from dynamics.base_dynamics import BaseDynamics
from utils.scaler import StandardScaler
from utils.logger import Logger


def get_log_prob(state, mean, std):
    cov_matrix = torch.diag_embed(std**2)
    distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
    
    state_log_prob = distribution.log_prob(state)[:]
    return state_log_prob


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        true_valnet: nn.Module,  # new
        model_valnet: nn.Module,  # new
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        tau: float = 0.003,
        gamma: float = 0.99,
        inc_var_loss: bool = True,
        use_weight_decay: bool = True,
        phi: float = 0.35,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.obs_scaler = deepcopy(scaler)
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self._tau = tau
        self._gamma = gamma
        self._phi = phi
        self.inc_var_loss = inc_var_loss
        self.use_weight_decay = use_weight_decay
        
        self.true_valnet, self.true_valnet_old = true_valnet, deepcopy(true_valnet)
        self.model_valnets = [deepcopy(model_valnet) for _ in range(self.model.num_ensemble)]
        self.model_valnet_olds = [deepcopy(model_valnet) for _ in range(self.model.num_ensemble)]
        
        self.true_val_optimizer = torch.optim.Adam(self.true_valnet.parameters(), 0.5e-5)
        
        for i, model_valnet in enumerate(self.model_valnets):
            if i == 0:
                model_valnet_params = list(model_valnet.parameters())
            else:
                model_valnet_params += list(model_valnet.parameters())
        self.model_val_optimizer = torch.optim.Adam(model_valnet_params, 0.5e-5)

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)  # !!! normalize obs
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs  # !!! add initial obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info

    def _sync_weight(self):
        for o, n in zip(self.true_valnet_old.parameters(), self.true_valnet.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        
        for i in range(self.model.num_ensemble):
            model_valnet_old = self.model_valnet_olds[i]
            model_valnet = self.model_valnets[i]
            for o, n in zip(model_valnet_old.parameters(), model_valnet.parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    @ torch.no_grad()
    def compute_model_uncertainty(self, obs: np.ndarray, action: np.ndarray, uncertainty_mode="aleatoric") -> np.ndarray:
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        if uncertainty_mode == "aleatoric":
            penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        elif uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean[:, :, :-1]
            next_obs_mean = np.mean(next_obses_mean, axis=0)
            diff = next_obses_mean - next_obs_mean
            penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        else:
            raise ValueError
        
        penalty = np.expand_dims(penalty, 1).astype(np.float32)

        return self._penalty_coef * penalty
    
    @ torch.no_grad()
    def predict_next_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act, device=obs_act.device)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        self.obs_length = obss.shape[-1]
        return inputs, targets, next_obss

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets, next_obs = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        init_train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        init_holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        train_next_obs, holdout_next_obs = next_obs[train_splits.indices], next_obs[holdout_splits.indices]

        self.scaler.fit(init_train_inputs)
        train_inputs = self.scaler.transform(init_train_inputs)
        holdout_inputs = self.scaler.transform(init_holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        
        self.obs_scaler.fit(next_obs)

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]
        
        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(
                init_train_inputs[data_idxes][..., :self.obs_length], train_next_obs[data_idxes], 
                train_inputs[data_idxes], train_targets[data_idxes], batch_size, 
                logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])
            
            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break
            # debug
            # break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(
        self,
        init_obs: np.ndarray,
        init_next_obs: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01,
        deterministic: bool = False,
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            obs_batch = init_obs[:, batch_num * batch_size:(batch_num+1) * batch_size]
            obs_batch = torch.as_tensor(obs_batch).to(self.model.device)
            reward_batch = init_obs[:, batch_num * batch_size:(batch_num+1) * batch_size][..., -1:]
            reward_batch = torch.as_tensor(reward_batch).to(self.model.device)
            next_obs_batch = init_next_obs[:, batch_num * batch_size:(batch_num+1) * batch_size]
            
            pred_diff_means, pred_diff_logvars = self.model(inputs_batch)
            inv_var = torch.exp(-pred_diff_logvars)
            
            # compute the true value
            true_value = self.true_valnet(obs_batch)
            true_next_value = self.true_valnet_old(next_obs_batch)
            true_value_target = reward_batch + self._gamma * true_next_value
            true_value_stable = self.true_valnet_old(obs_batch) # for updating model
            
            # compute the value learned from the model
            ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach()
            if deterministic:
                pred_means = pred_diff_means
            else:
                normal_random = torch.normal(mean=0.0, std=1.0, size=pred_diff_means.shape).to(self.model.device)
                pred_means = pred_diff_means + normal_random * ensemble_model_stds
                
            # obs_batch_tile = np.tile(obs_batch, (self.model.num_ensemble, 1, 1))
            pred_next_obs = obs_batch + pred_diff_means[..., :-1]
            pred_next_values = [net(ip) for ip, net in zip(torch.unbind(pred_next_obs), self.model_valnet_olds)]
            pred_next_value = torch.stack(pred_next_values)
            pred_value_target = pred_diff_means[..., -1:] + self._gamma * pred_next_value
            pred_values = [net(ip) for ip, net in zip(torch.unbind(obs_batch), self.model_valnets)]
            pred_value = torch.stack(pred_values)
            pred_values_stable = [net(ip) for ip, net in zip(torch.unbind(obs_batch), self.model_valnet_olds)]
            pred_value_stable = torch.stack(pred_values_stable)
            
            log_prob = get_log_prob(pred_diff_means, pred_diff_means, ensemble_model_stds).unsqueeze(-1)
            
            true_value_loss, model_value_loss = self.valnet_loss(
                true_value, true_value_target, pred_value, pred_value_target)

            # Compute true value networks' loss
            true_valnet_loss = torch.sum(true_value_loss)
            # Update parameters
            self.true_val_optimizer.zero_grad()
            true_valnet_loss.backward()
            self.true_val_optimizer.step()
        
            model_valnet_loss = torch.sum(model_value_loss)
            # Update parameters
            self.model_val_optimizer.zero_grad()
            model_valnet_loss.backward(retain_graph=True)
            self.model_val_optimizer.step()

            # compute training loss
            # groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1).to(util.device)
            (
                train_mse_losses, 
                train_var_losses, 
                V_loss
            ) = self.model_loss(
                pred_diff_means, pred_diff_logvars, targets_batch, true_value_stable, true_value_target, 
                pred_value_stable, pred_value_target, log_prob)
            
            train_mse_loss = torch.sum(train_mse_losses)
            train_var_loss = torch.sum(train_var_losses)
            train_val_loss = torch.sum(V_loss)
            train_transition_loss = train_mse_loss + train_var_loss + self._phi * train_val_loss + self.model.get_decay_loss()
            train_transition_loss += logvar_loss_coef * torch.sum(self.model.max_logvar) - logvar_loss_coef * torch.sum(self.model.min_logvar)  # why
            if self.use_weight_decay:
                decay_loss = self.model.get_decay_loss()
                train_transition_loss += decay_loss
            else:
                decay_loss = None
                
            # update transition model
            self.optim.zero_grad()
            train_transition_loss.backward()
            self.optim.step()
            
            # # Average over batch and dim, sum over ensembles.
            # mse_loss_inv = (torch.pow(pred_diff_means - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            # var_loss = pred_diff_logvars.mean(dim=(1, 2))
            # loss = mse_loss_inv.sum() + var_loss.sum()
            # loss = loss + self.model.get_decay_loss()
            # loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            # self.optim.zero_grad()
            # loss.backward()
            # self.optim.step()

            losses.append(train_transition_loss.item())
        return np.mean(losses)
    
    def valnet_loss(self, true_value, true_value_target, pred_value, pred_value_target):
        # Compute the loss of true value networks
        # true_value_loss = torch.mean(F.relu(true_value_target.detach() - true_value, inplace=False)**2)
        true_value_loss = torch.mean(torch.mean((true_value_target.detach() - true_value)**2, dim=-1), dim=-1)
        # Average over batch and dim, sum over ensembles.
        model_value_loss = torch.mean(torch.mean((pred_value - pred_value_target.detach())**2, dim=-1), dim=-1)
            
        return true_value_loss, model_value_loss
    
    def model_loss(
        self, pred_means, pred_logvars, groundtruths, true_value=None, true_value_target=None, 
        pred_value=None, pred_value_target=None, log_prob=None, mse_only=False,
        ):
        if self.inc_var_loss and not mse_only:
            # Average over batch and dim, sum over ensembles.
            inv_var = torch.exp(-pred_logvars)
            mse_losses = torch.mean(torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        elif mse_only:
            mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None
        else:
            assert 0
        
        if true_value != None:
            # true_value = true_value.view(1, -1, 1).tile((self.model.num_ensemble, 1, 1))
            # true_value_tile = true_value_target.view(1, -1, 1).tile((self.model.num_ensemble, 1, 1))
            
            advantage = (true_value - pred_value) * pred_value_target
            advantage = advantage.detach()
            V_loss = -advantage * log_prob
            V_loss = torch.mean(torch.mean(V_loss, dim=-1), dim=-1)

            # V_loss = torch.mean(torch.mean((true_value_tile.detach() - pred_value_target)**2, dim=-1), dim=-1)
            
            # true_value = torch.mean(torch.mean(torch.abs(true_value_tile.detach()), dim=-1), dim=-1).sum()
            # pred_value = torch.mean(torch.mean(torch.abs(pred_value_target.detach()), dim=-1), dim=-1).sum()
            # print("true value: ", true_value.item())
            # print("pred_value: ", pred_value.item())
            
            return mse_losses, var_losses, V_loss
        
        elif true_value == None:
            return mse_losses, var_losses

    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
