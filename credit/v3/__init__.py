"""CREDIT revision 3: randomized factorial effects and certified bank changes.

The neural outcome module requires the optional [neural] dependency.
Legacy implementations and their frozen result artifacts remain unchanged.
"""
from .drift import Level, TransportGate, transfer_level
from .factorial import admitted_pairs, estimate, regular_fraction
from .inference import certify, ppi, tune_lambda
from .types import Context, Episode, Memory, Node, NodeObservation, Role, SideEffect

__all__ = ["Level", "TransportGate", "transfer_level", "admitted_pairs", "estimate",
           "regular_fraction", "certify", "ppi", "tune_lambda", "Context", "Episode",
           "Memory", "Node", "NodeObservation", "Role", "SideEffect"]
