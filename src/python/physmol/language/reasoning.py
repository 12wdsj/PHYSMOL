"""ReasoningEngine: Causal reasoning, counterfactuals, and action planning.

Uses the causal graph and LGNN physics engine to answer questions like:
  - "What happens if I drop the ball?" -> simulate trajectory
  - "What if the mass doubled?" -> counterfactual simulation
  - "Push the block to the top" -> action sequence planning
  - "Explain elasticity" -> decompose concept from VSA
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ..vsa_store import RecipeStore, ConceptSynthesizer
from .abstract_reasoning import AbstractConceptReasoner


class CausalQuery:
    """A structured causal reasoning query."""

    def __init__(self, query_type: str, subject: str = "",
                 conditions: Optional[Dict[str, Any]] = None,
                 parameters: Optional[Dict[str, float]] = None):
        self.query_type = query_type  # "prediction", "counterfactual", "action", "explanation"
        self.subject = subject
        self.conditions = conditions or {}
        self.parameters = parameters or {}


class ReasoningEngine:
    """Execute causal reasoning queries against PHYSMOL's knowledge.

    Integrates:
      - Recipe store (what objects exist and their properties)
      - Causal graph (how events relate)
      - Physics knowledge (how forces, energy, motion work)
    """

    def __init__(self, recipe_store: RecipeStore,
                 causal_graph=None, lgnn=None):
        self.recipe_store = recipe_store
        self.synthesizer = ConceptSynthesizer(recipe_store)
        self.causal_graph = causal_graph
        self.lgnn = lgnn
        self.abstract_reasoner = AbstractConceptReasoner()

        # Physics knowledge base (rules derived from experience)
        self._physics_rules = self._init_physics_rules()

    def _init_physics_rules(self) -> Dict[str, dict]:
        """Initialize basic physics knowledge rules."""
        return {
            "gravity": {
                "description": "Objects fall downward under gravity",
                "trigger": ["fall", "drop", "gravity"],
                "effect": "Objects accelerate downward at 9.8 m/s^2",
                "conditions": {"supported": False},
            },
            "elasticity": {
                "description": "Elastic objects bounce back after impact",
                "trigger": ["bounce", "elastic", "rubber"],
                "effect": "Energy is conserved, object rebounds with reduced velocity",
                "conditions": {"elasticity": "elastic"},
            },
            "friction": {
                "description": "Friction opposes sliding motion",
                "trigger": ["slide", "friction", "rough"],
                "effect": "Sliding objects decelerate and eventually stop",
                "conditions": {"surface": "rough"},
            },
            "collision": {
                "description": "Conservation of momentum in collisions",
                "trigger": ["collide", "crash", "hit"],
                "effect": "Total momentum is conserved; energy may be lost",
                "conditions": {},
            },
            "inertia": {
                "description": "Objects at rest stay at rest, objects in motion stay in motion",
                "trigger": ["inertia", "mass", "heavy"],
                "effect": "Heavier objects require more force to accelerate",
                "conditions": {},
            },
            "potential_energy": {
                "description": "Elevated objects have gravitational potential energy",
                "trigger": ["height", "elevated", "above", "top", "ramp"],
                "effect": "EPE = mgh; converts to kinetic energy when falling",
                "conditions": {},
            },
        }

    def predict(self, subject: str, context: Optional[Dict] = None) -> dict:
        """Predict what happens to an object.

        Args:
            subject: object description (e.g., "ball", "red sphere")
            context: optional context (e.g., {"action": "drop", "height": 2.0})

        Returns: prediction dict with physics reasoning
        """
        context = context or {}
        action = context.get("action", "unknown")

        # Find matching objects
        matches = self.recipe_store.resonate(
            self._encode_query(subject), top_k=3)

        # Determine applicable physics rules
        applicable_rules = []
        subject_attrs = self._extract_attributes(subject)

        for rule_name, rule in self._physics_rules.items():
            for trigger in rule["trigger"]:
                if trigger in subject.lower() or trigger in action.lower():
                    applicable_rules.append(rule)
                    break

        # Build prediction
        prediction = {
            "subject": subject,
            "action": action,
            "matched_objects": matches[:3],
            "applicable_rules": [r["description"] for r in applicable_rules],
            "prediction": self._generate_prediction_text(subject, action, applicable_rules, context),
        }

        return prediction

    def counterfactual(self, subject: str, change: str,
                       context: Optional[Dict] = None) -> dict:
        """Reason about a counterfactual scenario.

        "What if the mass doubled?" -> heavier, falls same speed (Galileo),
        but hits harder (more momentum), more inertia.

        Args:
            subject: what we're reasoning about
            change: what property changes
        """
        context = context or {}

        # Map common counterfactuals to physics reasoning
        cf_rules = {
            "mass": {
                "heavier": "More inertia, harder to accelerate, same fall speed, greater impact force",
                "lighter": "Less inertia, easier to accelerate, same fall speed, smaller impact force",
                "double": "Twice the momentum at same velocity, twice the gravitational force, same acceleration",
            },
            "elasticity": {
                "more_elastic": "Higher bounce coefficient, less energy lost per bounce",
                "less_elastic": "Lower bounce, more energy lost as heat/sound",
                "rigid": "No bounce, all kinetic energy converted to deformation/sound",
            },
            "shape": {
                "sphere": "Rolls on inclines, minimal friction, smooth trajectory",
                "cube": "Slides or tumbles, higher friction on flat faces, irregular trajectory",
                "cylinder": "Rolls in one axis, slides in another",
            },
            "gravity": {
                "stronger": "Faster acceleration, higher impact velocity, shorter fall time",
                "weaker": "Slower acceleration, lower impact velocity, longer fall time",
                "zero": "No falling, objects float, no weight",
            },
            "friction": {
                "more": "Slower sliding, quicker stopping, more heat generated",
                "less": "Faster sliding, longer before stopping, less heat",
                "zero": "Perpetual sliding, no stopping, no heat from friction",
            },
        }

        # Determine what changed
        change_lower = change.lower()
        reasoning = "No specific counterfactual rule matched."

        for prop, scenarios in cf_rules.items():
            if prop in change_lower:
                for scenario, explanation in scenarios.items():
                    if scenario in change_lower or any(w in change_lower for w in scenario.split("_")):
                        reasoning = explanation
                        break
                break

        # If no specific rule, generate general reasoning
        if reasoning == "No specific counterfactual rule matched.":
            reasoning = f"If {change} changed, the physical behavior of {subject} would be affected according to the relevant conservation laws and force relationships."

        return {
            "subject": subject,
            "change": change,
            "reasoning": reasoning,
            "prediction": f"If {change} were different for {subject}: {reasoning}",
        }

    def explain_concept(self, concept: str) -> dict:
        """Explain a physical concept by decomposing from VSA knowledge.

        Args:
            concept: the concept to explain (e.g., "elasticity", "momentum")
        """
        # Map concepts to explanations
        concept_explanations = {
            "elasticity": {
                "definition": "Elasticity is the ability of a material to return to its original shape after being deformed.",
                "physics": "Elastic objects store potential energy during deformation and release it as kinetic energy during recovery. Governed by Hooke's law: F = -kx.",
                "examples": ["Rubber ball bouncing", "Spring stretching and contracting", "Trampoline rebounding"],
                "vsa_attrs": ["elasticity_elastic", "material_rubber"],
            },
            "gravity": {
                "definition": "Gravity is the force that attracts objects toward each other.",
                "physics": "Near Earth's surface, gravity accelerates objects at g = 9.8 m/s^2. Gravitational force: F = mg. Gravitational PE: E = mgh.",
                "examples": ["Ball falling to the ground", "Pendulum swinging", "Water flowing downhill"],
                "vsa_attrs": ["mass_medium", "mass_heavy"],
            },
            "friction": {
                "definition": "Friction is the force that opposes relative motion between surfaces in contact.",
                "physics": "Friction force: F = muN, where mu is the coefficient of friction and N is the normal force. Converts kinetic energy to heat.",
                "examples": ["Brake pads slowing a wheel", "Rubber on pavement", "Sandpaper smoothing wood"],
                "vsa_attrs": ["texture_rough", "texture_smooth"],
            },
            "momentum": {
                "definition": "Momentum is the product of mass and velocity: p = mv.",
                "physics": "Momentum is conserved in all collisions. Force is the rate of change of momentum: F = dp/dt.",
                "examples": ["Billiard ball collision", "Rocket propulsion", "Catching a ball"],
                "vsa_attrs": ["mass_heavy", "mass_light"],
            },
            "energy": {
                "definition": "Energy is the capacity to do work or cause change.",
                "physics": "Energy is conserved: it cannot be created or destroyed, only transformed. KE = 1/2mv^2, PE = mgh.",
                "examples": ["Roller coaster converting PE to KE", "Battery powering a motor", "Food fueling muscles"],
                "vsa_attrs": [],
            },
            "inertia": {
                "definition": "Inertia is the tendency of an object to resist changes in its state of motion.",
                "physics": "Quantified by mass. Newton's First Law: an object at rest stays at rest, an object in motion stays in motion, unless acted upon by a force.",
                "examples": ["Heavy ball harder to push", "Passengers lurching forward when bus stops", "Spacecraft coasting in vacuum"],
                "vsa_attrs": ["mass_heavy"],
            },
            "force": {
                "definition": "Force is an interaction that changes the motion of an object.",
                "physics": "Newton's Second Law: F = ma. Force is measured in Newtons (N = kg*m/s^2).",
                "examples": ["Pushing a block", "Gravity pulling a ball down", "Spring pushing back"],
                "vsa_attrs": ["elasticity_stiff", "mass_medium"],
            },
        }

        concept_lower = concept.lower()
        explanation = None
        for key, val in concept_explanations.items():
            if key in concept_lower or concept_lower in key:
                explanation = val
                break

        if explanation is None:
            explanation = {
                "definition": f"'{concept}' is a concept in the PHYSMOL knowledge base.",
                "physics": "Further physical exploration is needed to fully understand this concept.",
                "examples": [],
                "vsa_attrs": [],
            }

        # Find related objects in recipe store
        related_objects = []
        if explanation["vsa_attrs"]:
            related_objects = self.recipe_store.find_by_attributes(
                explanation["vsa_attrs"], top_k=5)

        return {
            "concept": concept,
            "explanation": explanation,
            "related_objects": related_objects,
        }

    def reason_abstract(self, text: str, max_depth: int = 4) -> dict:
        """Run multi-hop reasoning over abstract concepts.

        This is used for concepts such as fairness, justice, democracy,
        freedom, and slavery, where useful behavior requires moving from a
        compact abstraction to concrete constraints.
        """
        return self.abstract_reasoner.infer(text, max_depth=max_depth).as_dict()

    def plan_action(self, command: str, context: Optional[Dict] = None) -> dict:
        """Plan an action sequence from a command.

        Args:
            command: natural language command (e.g., "push the block to the top")
        """
        context = context or {}

        # Simple action decomposition
        action_verbs = {
            "push": {"force": "horizontal", "direction": "specified"},
            "pull": {"force": "horizontal_toward_agent", "direction": "toward"},
            "lift": {"force": "vertical_upward", "direction": "up"},
            "drop": {"force": "none_release", "direction": "down"},
            "place": {"force": "gentle_downward", "direction": "specified"},
            "move": {"force": "variable", "direction": "specified"},
            "stop": {"force": "friction/opposing", "direction": "opposing_motion"},
            "roll": {"force": "tangential", "direction": "along_surface"},
            "stack": {"force": "vertical_placement", "direction": "up"},
        }

        # Extract action verb
        tokens = command.lower().split()
        detected_action = None
        for token in tokens:
            if token in action_verbs:
                detected_action = token
                break

        if detected_action is None:
            detected_action = "move"  # default

        # Extract target object (simple heuristic)
        target = "object"
        for token in tokens:
            if token in ["ball", "cube", "block", "cylinder", "sphere", "box"]:
                target = token
                break

        # Extract destination
        destination = "specified location"
        dest_keywords = {
            "top": "top/ramp summit",
            "bottom": "bottom/ground level",
            "left": "left side",
            "right": "right side",
            "center": "center/middle",
            "ramp": "ramp",
            "table": "table surface",
        }
        for token in tokens:
            if token in dest_keywords:
                destination = dest_keywords[token]
                break

        action_info = action_verbs[detected_action]

        return {
            "command": command,
            "action": detected_action,
            "target": target,
            "destination": destination,
            "force_type": action_info["force"],
            "direction": action_info["direction"],
            "plan": [
                f"1. Identify the {target} in the scene",
                f"2. Compute required {action_info['force']} force",
                f"3. Apply force in {action_info['direction']} direction",
                f"4. Monitor trajectory until {destination} is reached",
                f"5. Verify final position",
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a text query into VSA space for resonance."""
        # Simple keyword-based vector composition
        from .text_encoder import TextToVSA
        encoder = TextToVSA(self.recipe_store.vsa_dim)
        return encoder.encode(text)

    def _extract_attributes(self, text: str) -> List[str]:
        """Extract attribute IDs from text keywords."""
        from .semantic_parser import SemanticParser
        from .text_encoder import TextToVSA
        encoder = TextToVSA(self.recipe_store.vsa_dim)
        parser = SemanticParser(encoder, self.recipe_store)
        tokens = encoder.tokenize(text)
        return parser.extract_attribute_hints(tokens)

    def _generate_prediction_text(self, subject: str, action: str,
                                  rules: list, context: dict) -> str:
        """Generate a natural language prediction."""
        if not rules:
            return f"The behavior of {subject} during '{action}' depends on its physical properties."

        rule_texts = [r["effect"] for r in rules]
        effects = "; ".join(rule_texts)

        if action in ["drop", "fall"]:
            return f"When {subject} is dropped: {effects}. It will accelerate downward and eventually impact the surface."
        elif action in ["push"]:
            return f"When {subject} is pushed: {effects}. It will accelerate in the direction of the applied force."
        elif action in ["collide", "hit", "crash"]:
            return f"When {subject} collides: {effects}. Momentum is transferred between objects."
        else:
            return f"For {subject} with action '{action}': {effects}"
