import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Optional, Any
import distrax

from loco_mujoco.algorithms.common.transformer_xl import TransformerXL


PHASE_HEAD_MODULE_NAME = "FullyConnectedNet_2"


class RandomActionDistribution:
    """Wrapper distribution that samples random actions from uniform [-1, 1] instead of the policy."""
    
    def __init__(self, base_distribution, action_dim):
        self.base_distribution = base_distribution
        self.action_dim = action_dim
        # Store batch shape from base distribution
        self._batch_shape = base_distribution.batch_shape
    
    def sample(self, seed, sample_shape=()):
        """Sample from uniform distribution [-1, 1] instead of the policy."""
        # Infer shape from base distribution's batch shape
        if len(sample_shape) == 0:
            if len(self._batch_shape) > 0:
                shape = self._batch_shape + (self.action_dim,)
            else:
                shape = (self.action_dim,)
        else:
            shape = sample_shape + (self.action_dim,)
        
        return jax.random.uniform(seed, minval=-0.25, maxval=0.25, shape=shape)
    
    def log_prob(self, value):
        """Return zero log prob for random actions."""
        # Return zeros with the same shape as value's batch dimensions
        if value.ndim > 1:
            # Batched case: return zeros for each sample in the batch
            return jnp.zeros(value.shape[:-1])
        else:
            # Single sample case
            return jnp.array(0.0)
    
    def entropy(self):
        """Return entropy of uniform distribution."""
        # Entropy of uniform [-1, 1] is log(2) = log(high - low)
        if len(self._batch_shape) > 0:
            return jnp.full(self._batch_shape, jnp.log(2.0))
        else:
            return jnp.array(jnp.log(2.0))
    
    @property
    def batch_shape(self):
        """Return batch shape from base distribution."""
        return self._batch_shape
    
    def __getattr__(self, name):
        """Delegate other attributes to base distribution."""
        return getattr(self.base_distribution, name)


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    actor_hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_hidden_layer_dims: Sequence[int] = (1024, 512)
    critic_obs_ind: jnp.ndarray = None
    phase_output_dim: int = 0
    phase_hidden_layer_dims: Sequence[int] = (256, 128)
    # random: bool = False

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        # self.actor_rms = RunningMeanStd()
        # self.critic_rms = RunningMeanStd()

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        # actor_x = self.actor_rms(actor_x)
        actor_mean = FullyConnectedNet(self.actor_hidden_layer_dims, self.action_dim, self.activation,
                                       None, False, False)(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        
        # # Wrap policy with random action distribution if random is True
        # if self.random:
        #     pi = RandomActionDistribution(pi, self.action_dim)

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        # critic_x = self.critic_rms(critic_x)
        critic = FullyConnectedNet(self.critic_hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)

        value = jnp.squeeze(critic, axis=-1)
        if self.phase_output_dim <= 0:
            return pi, value

        phase = FullyConnectedNet(
            self.phase_hidden_layer_dims,
            self.phase_output_dim,
            self.activation,
            "sigmoid",
            False,
            False,
            name=PHASE_HEAD_MODULE_NAME,
        )(jax.lax.stop_gradient(actor_x))
        phase = jnp.squeeze(phase, axis=-1)
        return pi, value, phase


class TransformerXLActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    actor_hidden_layer_dims: Sequence[int] = (256, 256)
    actor_obs_ind: jnp.ndarray = None
    critic_hidden_layer_dims: Sequence[int] = (256, 256)
    critic_obs_ind: jnp.ndarray = None
    model_dim: int = 128
    n_layers: int = 2
    n_heads: int = 2
    ff_dim: int = 256
    dropout: float = 0.0
    mem_len: int = 16
    positional_encoding: str = "absolute"
    phase_output_dim: int = 0
    phase_hidden_layer_dims: Sequence[int] = (256, 128)

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.actor_rms = RunningMeanStd()
        self.actor_embed = nn.Dense(
            self.model_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_trxl = TransformerXL(
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            mem_len=self.mem_len,
            positional_encoding=self.positional_encoding,
        )

    def _select_obs(self, x: jnp.ndarray, obs_ind: Optional[jnp.ndarray]) -> jnp.ndarray:
        return x if obs_ind is None else x[..., obs_ind]

    def _normalize_and_embed(self, x: jnp.ndarray, rms: Any, embed: nn.Dense) -> jnp.ndarray:
        if x.ndim == 1:
            x = x[None, None, ...]
        elif x.ndim == 2:
            x = x[None, ...]
        x_flat = x.reshape((-1, x.shape[-1]))
        x_norm = rms(x_flat)
        x_norm = x_norm.reshape(x.shape)
        h = embed(x_norm)
        return self.activation_fn(h)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 mems: Optional[Any] = None,
                 attn_mask: Optional[jnp.ndarray] = None,
                 train: bool = False):
        mems_actor = mems
        actor_x = self._select_obs(x, self.actor_obs_ind)
        critic_x = self._select_obs(x, self.critic_obs_ind)
        actor_h = self._normalize_and_embed(actor_x, self.actor_rms, self.actor_embed)
        actor_h, new_mems, layer_h = self.actor_trxl(actor_h, mems_actor, attn_mask, train)
        h_last = actor_h[-1]
        layer_h_last = [h[-1] for h in layer_h]
        actor_mean = FullyConnectedNet(
            self.actor_hidden_layer_dims, self.action_dim, self.activation, None, False, False
        )(h_last)
        critic = FullyConnectedNet(
            self.critic_hidden_layer_dims, 1, self.activation, None, False, False
        )(critic_x)

        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        value = jnp.squeeze(critic, axis=-1)
        if self.phase_output_dim <= 0:
            return pi, value, new_mems, layer_h_last

        phase = FullyConnectedNet(
            self.phase_hidden_layer_dims,
            self.phase_output_dim,
            self.activation,
            "sigmoid",
            False,
            False,
            name=PHASE_HEAD_MODULE_NAME,
        )(jax.lax.stop_gradient(h_last))
        phase = jnp.squeeze(phase, axis=-1)
        return pi, value, phase, new_mems, layer_h_last


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
        batch_count = x.shape[0]

        if self.is_mutable_collection("run_stats"):
            # Update counts
            updated_count = count.value + batch_count

            # Numerically stable mean and variance update
            delta = batch_mean - mean.value
            new_mean = mean.value + delta * batch_count / updated_count

            # Compute the new variance using Welford's method
            m_a = var.value * count.value
            m_b = batch_var * batch_count
            M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
            new_var = M2 / updated_count

            # Update state variables
            mean.value = new_mean
            var.value = new_var
            count.value = updated_count
        else:
            new_mean = mean.value
            new_var = var.value

        # Normalize input
        normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

        return jnp.squeeze(normalized_x)
