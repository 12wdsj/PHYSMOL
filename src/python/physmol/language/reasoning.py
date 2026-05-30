"""ReasoningEngine: Causal reasoning, counterfactuals, action planning, and code reasoning.

Uses the causal graph and LGNN physics engine to answer questions like:
  - "What happens if I drop the ball?" -> simulate trajectory
  - "What if the mass doubled?" -> counterfactual simulation
  - "Push the block to the top" -> action sequence planning
  - "Explain elasticity" -> decompose concept from VSA
  - "Explain quicksort" -> algorithm explanation with complexity
  - "Compare BFS vs DFS" -> algorithm comparison
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
    # Code reasoning (unified with physical reasoning via VSA)
    # ------------------------------------------------------------------

    def explain_code_concept(self, concept: str) -> dict:
        """Explain a code/algorithm concept using VSA knowledge.

        This mirrors explain_concept() but for programming concepts.
        Both physical and code concepts exist in the same VSA space.
        """
        code_explanations = {
            "quicksort": {
                "definition": "A divide-and-conquer sorting algorithm that partitions around a pivot",
                "algorithm": "Choose pivot, partition into smaller/larger, recurse on both halves",
                "complexity": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n^2)"},
                "vsa_attrs": ["algorithm_sort", "algorithm_divide", "algorithm_conquer"],
                "use_cases": ["general purpose sorting", "in-place sorting"],
            },
            "merge sort": {
                "definition": "A stable, divide-and-conquer sorting algorithm that divides the array in half, sorts each half, and merges them",
                "algorithm": "Divide array in half, recursively sort both halves, merge sorted halves",
                "complexity": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)"},
                "vsa_attrs": ["algorithm_sort", "algorithm_divide", "algorithm_merge"],
                "use_cases": ["stable sorting", "external sorting", "linked list sorting"],
            },
            "heap sort": {
                "definition": "A comparison-based sorting algorithm using a binary heap data structure",
                "algorithm": "Build max heap, extract max repeatedly to get sorted array",
                "complexity": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)"},
                "vsa_attrs": ["algorithm_sort", "data_structure_heap"],
                "use_cases": ["in-place sorting", "guaranteed O(n log n)"],
            },
            "binary search": {
                "definition": "A search algorithm that finds an element in a sorted array by repeatedly dividing the search interval in half",
                "algorithm": "Compare with middle element, eliminate half, repeat",
                "complexity": {"best": "O(1)", "average": "O(log n)", "worst": "O(log n)"},
                "vsa_attrs": ["algorithm_search", "algorithm_divide"],
                "use_cases": ["searching sorted data", "finding boundaries"],
            },
            "bfs": {
                "definition": "Breadth-First Search explores nodes level by level using a queue",
                "algorithm": "Start at root, visit all neighbors, then neighbors' neighbors",
                "complexity": {"time": "O(V + E)", "space": "O(V)"},
                "vsa_attrs": ["algorithm_traverse", "data_structure_queue", "data_structure_graph"],
                "use_cases": ["shortest path in unweighted graph", "level-order traversal"],
            },
            "dfs": {
                "definition": "Depth-First Search explores as far as possible along each branch before backtracking",
                "algorithm": "Start at root, go deep, backtrack when stuck",
                "complexity": {"time": "O(V + E)", "space": "O(V)"},
                "vsa_attrs": ["algorithm_traverse", "algorithm_backtrack", "data_structure_graph"],
                "use_cases": ["topological sort", "cycle detection", "path finding"],
            },
            "dynamic programming": {
                "definition": "Optimization technique that solves complex problems by breaking them into overlapping subproblems",
                "algorithm": "Identify subproblems, solve bottom-up or top-down with memoization",
                "complexity": {"time": "varies", "space": "O(n) to O(n^2)"},
                "vsa_attrs": ["algorithm_dynamic", "algorithm_divide"],
                "use_cases": ["optimization problems", "sequence alignment", "knapsack"],
            },
            "linked list": {
                "definition": "A linear data structure where elements are stored in nodes linked by pointers",
                "algorithm": "Each node contains data and a pointer to the next node",
                "complexity": {"access": "O(n)", "search": "O(n)", "insert": "O(1)", "delete": "O(1)"},
                "vsa_attrs": ["data_structure_linked_list", "data_structure_list"],
                "use_cases": ["dynamic size", "frequent insert/delete", "implementation of stacks/queues"],
            },
            "stack": {
                "definition": "A LIFO (Last-In-First-Out) data structure",
                "algorithm": "Push to add, Pop to remove from top",
                "complexity": {"push": "O(1)", "pop": "O(1)", "peek": "O(1)"},
                "vsa_attrs": ["data_structure_stack", "operation_insert", "operation_remove"],
                "use_cases": ["function calls", "undo operations", "expression evaluation"],
            },
            "queue": {
                "definition": "A FIFO (First-In-First-Out) data structure",
                "algorithm": "Enqueue to add, Dequeue to remove from front",
                "complexity": {"enqueue": "O(1)", "dequeue": "O(1)"},
                "vsa_attrs": ["data_structure_queue", "data_structure_deque"],
                "use_cases": ["BFS", "task scheduling", "buffer"],
            },
            "graph": {
                "definition": "A collection of nodes (vertices) connected by edges",
                "algorithm": "Represented as adjacency list or adjacency matrix",
                "complexity": {"adjacency_list": "O(V+E) space", "adjacency_matrix": "O(V^2) space"},
                "vsa_attrs": ["data_structure_graph", "data_structure_tree"],
                "use_cases": ["networks", "social connections", "pathfinding"],
            },
            "recursion": {
                "definition": "A technique where a function calls itself to solve smaller subproblems",
                "algorithm": "Base case + recursive case",
                "complexity": {"time": "varies", "space": "O(depth) for call stack"},
                "vsa_attrs": ["control_flow_recursion", "algorithm_recurse"],
                "use_cases": ["tree traversal", "divide and conquer", "backtracking"],
            },
            "dijkstra": {
                "definition": "A greedy algorithm that finds the shortest path from a source node to all other nodes in a weighted graph",
                "algorithm": "Use a priority queue to always process the closest unvisited node",
                "complexity": {"time": "O((V + E) log V)", "space": "O(V)"},
                "vsa_attrs": ["algorithm_greedy", "data_structure_graph", "data_structure_heap"],
                "use_cases": ["shortest path in weighted graph", "network routing", "GPS navigation"],
            },
            "binary tree": {
                "definition": "A hierarchical data structure where each node has at most two children",
                "algorithm": "Nodes organized by value: left < parent < right for BST",
                "complexity": {"access": "O(log n) avg", "search": "O(log n) avg", "insert": "O(log n) avg"},
                "vsa_attrs": ["data_structure_tree", "data_structure_graph"],
                "use_cases": ["binary search trees", "expression trees", "heaps"],
            },
            "hash map": {
                "definition": "A data structure that maps keys to values using a hash function",
                "algorithm": "Hash key to find bucket, handle collisions with chaining or open addressing",
                "complexity": {"get": "O(1) avg", "put": "O(1) avg", "delete": "O(1) avg"},
                "vsa_attrs": ["data_structure_hash", "operation_find"],
                "use_cases": ["caching", "counting", "lookup tables"],
            },
            "lru cache": {
                "definition": "A cache that evicts the least recently used item when full",
                "algorithm": "Combine hash map for O(1) access with doubly-linked list for O(1) reorder",
                "complexity": {"get": "O(1)", "put": "O(1)"},
                "vsa_attrs": ["data_structure_hash", "data_structure_linked_list"],
                "use_cases": ["page replacement", "database caching", "web caching"],
            },
        }

        concept_lower = concept.lower()
        explanation = None
        for key, val in code_explanations.items():
            if key in concept_lower or concept_lower in key:
                explanation = val
                break

        if explanation is None:
            explanation = {
                "definition": f"'{concept}' is a programming concept in the PHYSMOL knowledge base.",
                "algorithm": "Further exploration is needed to fully understand this concept.",
                "complexity": {},
                "vsa_attrs": [],
                "use_cases": [],
            }

        # Find related objects/code patterns in recipe store
        related_patterns = []
        if explanation.get("vsa_attrs"):
            related_patterns = self.recipe_store.find_by_attributes(
                explanation["vsa_attrs"], top_k=5)

        return {
            "concept": concept,
            "kind": "code_explanation",
            "explanation": explanation,
            "related_patterns": related_patterns,
        }

    def compare_algorithms(self, algo1: str, algo2: str) -> dict:
        """Compare two algorithms using VSA concept decomposition.

        Uses the unified VSA space to compare physical and code concepts alike.
        """
        exp1 = self.explain_code_concept(algo1)
        exp2 = self.explain_code_concept(algo2)

        comparison = {
            "algorithms": [algo1, algo2],
            "kind": "algorithm_comparison",
            algo1: exp1.get("explanation", {}),
            algo2: exp2.get("explanation", {}),
            "comparison_points": [],
        }

        # Compare complexity
        c1 = exp1.get("explanation", {}).get("complexity", {})
        c2 = exp2.get("explanation", {}).get("complexity", {})
        if c1 and c2:
            comparison["comparison_points"].append({
                "aspect": "complexity",
                algo1: c1,
                algo2: c2,
            })

        # Compare use cases
        u1 = exp1.get("explanation", {}).get("use_cases", [])
        u2 = exp2.get("explanation", {}).get("use_cases", [])
        if u1 and u2:
            comparison["comparison_points"].append({
                "aspect": "use_cases",
                algo1: u1,
                algo2: u2,
            })

        # Compare VSA attributes (shared concepts)
        attrs1 = set(exp1.get("explanation", {}).get("vsa_attrs", []))
        attrs2 = set(exp2.get("explanation", {}).get("vsa_attrs", []))
        shared = attrs1.intersection(attrs2)
        if shared:
            comparison["shared_concepts"] = list(shared)

        return comparison

    def reason_about_code(self, task: str) -> dict:
        """Reason about a coding task using VSA concepts.

        This is the code equivalent of predict() - given a task description,
        determine what algorithms/data structures are needed.
        """
        task_lower = task.lower()

        # Detect algorithm type
        algorithm_hints = {
            "sort": ["sort", "order", "arrange", "排序"],
            "search": ["search", "find", "look for", "查找", "搜索"],
            "traverse": ["traverse", "visit", "遍历"],
            "graph": ["graph", "node", "edge", "图", "节点"],
            "tree": ["tree", "binary tree", "树"],
            "list": ["list", "array", "链表", "数组"],
            "stack": ["stack", "push", "pop", "栈"],
            "queue": ["queue", "enqueue", "dequeue", "队列"],
        }

        detected_algorithms = []
        for algo, hints in algorithm_hints.items():
            if any(h in task_lower for h in hints):
                detected_algorithms.append(algo)

        # Detect data structure
        ds_hints = {
            "array": ["array", "list", "数组"],
            "linked_list": ["linked list", "链表"],
            "stack": ["stack", "栈"],
            "queue": ["queue", "队列"],
            "graph": ["graph", "图"],
            "tree": ["tree", "树"],
            "hash": ["hash", "dictionary", "map", "哈希"],
        }

        detected_ds = []
        for ds, hints in ds_hints.items():
            if any(h in task_lower for h in hints):
                detected_ds.append(ds)

        # Build reasoning result
        result = {
            "kind": "code_reasoning",
            "task": task,
            "detected_algorithms": detected_algorithms,
            "detected_data_structures": detected_ds,
            "suggestions": [],
        }

        # Generate suggestions based on detected patterns
        if "sort" in detected_algorithms:
            result["suggestions"].append({
                "algorithm": "quicksort",
                "reason": "Efficient general-purpose sort, O(n log n) average",
                "vsa_attrs": ["algorithm_sort", "algorithm_divide"],
            })
        if "search" in detected_algorithms:
            if "sorted" in task_lower or "order" in task_lower:
                result["suggestions"].append({
                    "algorithm": "binary_search",
                    "reason": "O(log n) search on sorted data",
                    "vsa_attrs": ["algorithm_search", "algorithm_divide"],
                })
            else:
                result["suggestions"].append({
                    "algorithm": "linear_search",
                    "reason": "O(n) search, works on any data",
                    "vsa_attrs": ["algorithm_search", "algorithm_iterate"],
                })
        if "graph" in detected_algorithms or "graph" in detected_ds:
            if "shortest" in task_lower or "level" in task_lower:
                result["suggestions"].append({
                    "algorithm": "bfs",
                    "reason": "BFS finds shortest path in unweighted graphs",
                    "vsa_attrs": ["algorithm_traverse", "data_structure_queue"],
                })
            else:
                result["suggestions"].append({
                    "algorithm": "dfs",
                    "reason": "DFS is simpler for path finding and cycle detection",
                    "vsa_attrs": ["algorithm_traverse", "algorithm_backtrack"],
                })

        return result

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
