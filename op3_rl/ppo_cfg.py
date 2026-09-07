"""RSL-RL PPO runner config for OP3.

Network/algorithm sizes follow Isaac Lab's Cassie flat-terrain config
(a comparable biped) as a reasonable starting point, not a tuned result.
`max_iterations`/`num_envs` are overridden by CLI args in both
scripts/train_ppo_smoke.py (tiny, for the Stage 3 smoke test) and
scripts/train_ppo.py (the real Stage 4 training/resume script) -- the
defaults below are just the fallback if run with no args.
"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class OP3FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "op3_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Revision (see STAGE4_PPO_TRAINING.md): Cassie's entropy_coef=0.01
        # caused runaway exploration on OP3 -- Mean action noise std climbed
        # monotonically (1.00 -> 1.38 over 500 iterations) instead of
        # shrinking as the policy converges, and episode length declined in
        # lockstep (a noisier policy falls faster). Dropped to 0.0: nothing
        # here needs extra entropy pressure to avoid premature convergence
        # given the reward signal is already weak/dominated by falling.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
