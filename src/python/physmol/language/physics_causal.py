"""Compositional physics causal model.

Each derived quantity is a monomial of base quantities:
    Q = k * m^a * g^b * v^c * h^d * mu^e * ...

A perturbation (base quantity multiplied by factor f) propagates:
    Q_new = Q * f^exponent

This lets non-obvious conclusions emerge naturally — e.g. Galileo's
insight that free-fall speed is independent of mass (exponent 0).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re


# ---------------------------------------------------------------------------
# Quantity definitions
# ---------------------------------------------------------------------------

@dataclass
class Quantity:
    name: str
    label: str          # human-readable name
    # exponents: base_qty_name -> exponent in monomial
    exponents: Dict[str, float]
    description: str    # physics formula / note


# Base quantities (index-0 exponent = 1 for themselves)
BASE = ["mass", "gravity", "velocity", "height", "friction_coeff",
        "elasticity_coeff", "force_applied", "area"]

# Derived quantities as monomials over base quantities
_QUANTITIES: List[Quantity] = [
    # Kinematics
    Quantity("fall_accel", "free-fall acceleration",
             {"gravity": 1, "mass": 0},
             "a = g  (independent of mass - Galileo)"),
    Quantity("fall_time", "time to fall from height h",
             {"height": 0.5, "gravity": -0.5, "mass": 0},
             "t = sqrt(2h/g)"),
    Quantity("impact_velocity", "velocity at impact",
             {"gravity": 0.5, "height": 0.5, "mass": 0},
             "v = sqrt(2gh)  (independent of mass)"),
    # Dynamics
    Quantity("weight", "gravitational weight",
             {"mass": 1, "gravity": 1},
             "W = mg"),
    Quantity("momentum", "linear momentum",
             {"mass": 1, "velocity": 1},
             "p = mv"),
    Quantity("kinetic_energy", "kinetic energy",
             {"mass": 1, "velocity": 2},
             "KE = 0.5*mv^2"),
    Quantity("impact_force", "impact force (impulse / contact time)",
             {"mass": 1, "gravity": 0.5, "height": 0.5},
             "F_impact ~ mv = m*sqrt(2gh)"),
    Quantity("potential_energy", "gravitational potential energy",
             {"mass": 1, "gravity": 1, "height": 1},
             "PE = mgh"),
    # Forces
    Quantity("net_force", "net force",
             {"mass": 1, "gravity": 1},
             "F_net = ma = mg (free fall)"),
    Quantity("friction_force", "friction force",
             {"friction_coeff": 1, "mass": 1, "gravity": 1},
             "F_f = u*mg"),
    Quantity("stopping_distance", "sliding stopping distance",
             {"velocity": 2, "friction_coeff": -1, "gravity": -1},
             "d = v^2/(2*u*g)"),
    # Bouncing
    Quantity("bounce_height", "height after elastic bounce",
             {"elasticity_coeff": 2, "height": 1},
             "h' = e^2*h  (e = restitution coefficient)"),
    Quantity("energy_lost_bounce", "energy lost per bounce",
             {"mass": 1, "gravity": 1, "height": 1, "elasticity_coeff": 0},
             "E_lost = (1-e^2)*mgh  (decreases with higher elasticity)"),
]

_QUANTITY_MAP: Dict[str, Quantity] = {q.name: q for q in _QUANTITIES}


# ---------------------------------------------------------------------------
# Natural language → perturbation parser
# ---------------------------------------------------------------------------

# Maps (keyword_pattern, base_quantity, scale_factor)
_CHANGE_RULES: List[Tuple[str, str, float]] = [
    # mass
    (r"heavier|more mass|mass.{0,8}increas|重|更重|质量增", "mass", 2.0),
    (r"lighter|less mass|mass.{0,8}decreas|更轻|质量减", "mass", 0.5),
    (r"mass.{0,5}doubl|质量翻倍|double.{0,5}mass", "mass", 2.0),
    (r"mass.{0,5}half|质量减半", "mass", 0.5),
    (r"massless|no mass", "mass", 0.01),
    # gravity
    (r"stronger gravity|gravity.{0,8}stronger|gravity.{0,8}increas|重力更强|更强的重力", "gravity", 2.0),
    (r"weaker gravity|gravity.{0,8}weaker|gravity.{0,8}decreas|重力更弱|更弱的重力", "gravity", 0.5),
    (r"no gravity|zero gravity|weightless|失重|无重力", "gravity", 0.001),
    (r"moon|月球", "gravity", 0.17),
    (r"jupiter|木星", "gravity", 2.53),
    # velocity / speed
    (r"faster|higher speed|speed.{0,8}increas|更快|速度更大", "velocity", 2.0),
    (r"slower|lower speed|speed.{0,8}decreas|更慢|速度更小", "velocity", 0.5),
    # height
    (r"higher|taller|greater height|更高", "height", 2.0),
    (r"lower|shorter height|更低", "height", 0.5),
    # friction
    (r"more friction|rougher|增大摩擦|更粗糙", "friction_coeff", 2.0),
    (r"less friction|smoother|减小摩擦|更光滑", "friction_coeff", 0.5),
    (r"no friction|frictionless|zero friction|无摩擦|光滑无摩擦", "friction_coeff", 0.001),
    # elasticity
    (r"more elastic|bouncier|弹性更大|更有弹性", "elasticity_coeff", 1.5),
    (r"less elastic|rigid|inelastic|弹性更小|刚性|无弹性", "elasticity_coeff", 0.1),
]

def parse_change(text: str) -> Optional[Tuple[str, float, str]]:
    """Return (base_quantity, scale_factor, matched_description) or None."""
    t = text.lower()
    for pattern, qty, factor in _CHANGE_RULES:
        if re.search(pattern, t):
            return qty, factor, pattern
    return None


# ---------------------------------------------------------------------------
# Causal propagation
# ---------------------------------------------------------------------------

@dataclass
class CausalEffect:
    quantity: str
    label: str
    factor: float       # how much it changes (relative to baseline)
    direction: str      # "increases", "decreases", "unchanged"
    formula: str


def _direction(factor: float) -> str:
    if factor > 1.05:
        return "increases"
    if factor < 0.95:
        return "decreases"
    return "unchanged"


def propagate(base_qty: str, scale: float) -> List[CausalEffect]:
    """Given a perturbation to base_qty by scale, return all derived effects."""
    effects = []
    for q in _QUANTITIES:
        exp = q.exponents.get(base_qty, 0.0)
        factor = scale ** exp
        effects.append(CausalEffect(
            quantity=q.name,
            label=q.label,
            factor=round(factor, 4),
            direction=_direction(factor),
            formula=q.description,
        ))
    return effects


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

_NOTABLE = [
    "fall_accel", "impact_velocity", "fall_time",
    "momentum", "impact_force", "kinetic_energy",
    "weight", "potential_energy",
    "friction_force", "stopping_distance",
    "bounce_height",
]

_LABELS = {
    "increases": "[+] increases",
    "decreases": "[-] decreases",
    "unchanged": "[=] unchanged",
}

def summarize(subject: str, base_qty: str, scale: float, effects: List[CausalEffect]) -> str:
    """Return a concise, readable counterfactual summary."""
    direction_word = "larger" if scale > 1 else "smaller"
    qty_names = {
        "mass": "mass",
        "gravity": "gravity",
        "velocity": "initial speed",
        "height": "drop height",
        "friction_coeff": "friction",
        "elasticity_coeff": "elasticity",
    }
    qty_label = qty_names.get(base_qty, base_qty)

    lines = [f"If the {subject}'s {qty_label} were {direction_word} (x{scale}):"]

    notable_effects = [e for e in effects if e.quantity in _NOTABLE]
    notable_effects.sort(key=lambda e: _NOTABLE.index(e.quantity))

    for e in notable_effects:
        symbol = _LABELS[e.direction]
        factor_str = f"x{e.factor:.2g}" if e.direction != "unchanged" else ""
        lines.append(f"  - {e.label}: {symbol} {factor_str}  [{e.formula}]")

    # Add the most physically interesting insight
    accel = next((e for e in effects if e.quantity == "fall_accel"), None)
    if accel and accel.direction == "unchanged" and base_qty == "mass":
        lines.append("\nKey insight: free-fall acceleration is independent of mass (Galileo, 1638).")

    if base_qty == "gravity" and scale < 0.01:
        lines.append("\nKey insight: without gravity, objects float — no weight, no falling.")

    if base_qty == "friction_coeff" and scale < 0.01:
        lines.append("\nKey insight: without friction, sliding objects never stop (Newton's 1st Law).")

    return "\n".join(lines)
