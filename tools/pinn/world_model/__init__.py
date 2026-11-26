"""K1_WM_01 — State Transition Dataset package.

Unified state-transition data interface over tools/pinn/stage1_data for the
three planned world-model variants:
  B. SuperMimic-style data-driven world model
  C. Free-Dynamics PINN + rollout
  D. Hybrid physics-informed world model

Components:
  context.py                  episode physical context xi_e (from the XML)
  wm_data.py                  StateTransitionDataset / RolloutWindowDataset
  build_transitions.py        materialize transitions + window indices + checks
  verify_state_integration.py MuJoCo position-integration verification
"""

from .context import episode_context
from .wm_data import StateTransitionDataset, RolloutWindowDataset

__all__ = ["episode_context", "StateTransitionDataset", "RolloutWindowDataset"]
