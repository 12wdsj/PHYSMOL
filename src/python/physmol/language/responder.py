"""Responder: Convert reasoning results into natural language responses.

Uses template-based generation for initial implementation.
Can be upgraded to use an external LLM for more natural responses.
"""

from typing import Dict, List, Optional, Any


class Responder:
    """Generate natural language responses from reasoning results.

    Templates are organized by intent type:
      - question -> answer templates
      - command -> acknowledgment + plan templates
      - explanation -> structured explanation templates
      - counterfactual -> hypothetical reasoning templates
    """

    def __init__(self):
        self._templates = self._init_templates()

    def _init_templates(self) -> Dict[str, dict]:
        return {
            "question": {
                "prediction": (
                    "Based on my physical understanding: {prediction}\n"
                    "{details}"
                ),
                "object_found": (
                    "I found {count} object(s) matching your description:\n"
                    "{object_list}"
                ),
                "no_match": (
                    "I couldn't find objects matching '{query}' in my knowledge base. "
                    "I may need to explore more to understand this."
                ),
            },
            "command": {
                "acknowledged": (
                    "Understood. I will {action} the {target}.\n\n"
                    "Plan:\n{plan}"
                ),
                "cannot_execute": (
                    "I understand you want to {action}, but I cannot execute "
                    "this command because: {reason}"
                ),
            },
            "explanation": {
                "concept": (
                    "**{concept}**\n\n"
                    "Definition: {definition}\n\n"
                    "Physics: {physics}\n\n"
                    "Examples:\n{examples}\n\n"
                    "{related}"
                ),
            },
            "counterfactual": {
                "reasoning": (
                    "Hypothetical scenario: {subject} with {change}\n\n"
                    "Reasoning: {reasoning}\n\n"
                    "Prediction: {prediction}"
                ),
            },
            "abstract": {
                "reasoning": (
                    "Abstract reasoning activated: {roots}\n\n"
                    "Reasoning chain:\n{chains}\n\n"
                    "Concrete implications:\n{applications}"
                ),
            },
            "social": {
                "state": (
                    "Theory-of-mind model for {agent}:\n"
                    "{answer}"
                ),
            },
            "conversation": {
                "reply": "{response}",
            },
            "memory": {
                "stored": "I stored that in long-term memory: {content}",
                "retrieved": "Relevant long-term memories:\n{records}",
            },
            "transfer": {
                "hypotheses": (
                    "Cross-domain transfer hypotheses from {source} to {target}:\n"
                    "{hypotheses}"
                ),
            },
            "abstract_task": {
                "result": (
                    "{domain} reasoning\n\n"
                    "Premises:\n{premises}\n\n"
                    "Rules:\n{rules}\n\n"
                    "Conclusion: {conclusion}"
                ),
            },
            "error": {
                "general": "I encountered an issue processing your request: {error}",
                "no_understanding": (
                    "I'm not sure I understand '{query}'. Could you rephrase? "
                    "I can answer questions about objects, explain physics concepts, "
                    "or execute physical commands."
                ),
            },
        }

    def respond(self, parsed_query: dict, reasoning_result: dict) -> str:
        """Generate a response from parsed query and reasoning result.

        Args:
            parsed_query: from SemanticParser.parse_query()
            reasoning_result: from ReasoningEngine method result

        Returns: natural language response string
        """
        intent = parsed_query.get("intent", "unknown")
        kind = reasoning_result.get("kind")

        if kind == "abstract_reasoning":
            return self._respond_abstract(parsed_query, reasoning_result)
        if kind == "theory_of_mind":
            return self._respond_social(parsed_query, reasoning_result)
        if kind == "conversation":
            return self._templates["conversation"]["reply"].format(
                response=reasoning_result.get("response", ""))
        if kind == "memory":
            return self._respond_memory(parsed_query, reasoning_result)
        if kind == "cross_domain_transfer":
            return self._respond_transfer(parsed_query, reasoning_result)
        if kind == "abstract_task":
            return self._respond_abstract_task(parsed_query, reasoning_result)

        if intent == "question":
            return self._respond_question(parsed_query, reasoning_result)
        elif intent == "command":
            return self._respond_command(parsed_query, reasoning_result)
        elif intent == "explanation":
            return self._respond_explanation(parsed_query, reasoning_result)
        elif intent == "counterfactual":
            return self._respond_counterfactual(parsed_query, reasoning_result)
        elif intent == "abstract":
            return self._respond_abstract(parsed_query, reasoning_result)
        elif intent == "social":
            return self._respond_social(parsed_query, reasoning_result)
        elif intent == "memory":
            return self._respond_memory(parsed_query, reasoning_result)
        elif intent == "transfer":
            return self._respond_transfer(parsed_query, reasoning_result)
        elif intent == "abstract_task":
            return self._respond_abstract_task(parsed_query, reasoning_result)
        else:
            return self._respond_unknown(parsed_query)

    def _respond_question(self, parsed: dict, result: dict) -> str:
        """Generate response for a question."""
        if "prediction" in result:
            details = ""
            if result.get("matched_objects"):
                obj_list = "\n".join(
                    f"  - {obj_id} (similarity: {score:.2f})"
                    for obj_id, score in result["matched_objects"]
                )
                details = f"Related objects:\n{obj_list}"
            if result.get("applicable_rules"):
                rules = "\n".join(f"  - {r}" for r in result["applicable_rules"])
                details += f"\nApplicable physics:\n{rules}"

            return self._templates["question"]["prediction"].format(
                prediction=result["prediction"],
                details=details
            )

        if "matched_objects" in result:
            matches = result["matched_objects"]
            if matches:
                obj_list = "\n".join(
                    f"  - {obj_id} (score: {score:.2f})"
                    for obj_id, score in matches
                )
                return self._templates["question"]["object_found"].format(
                    count=len(matches), object_list=obj_list
                )

        return self._templates["question"]["no_match"].format(
            query=parsed.get("text", ""))

    def _respond_command(self, parsed: dict, result: dict) -> str:
        """Generate response for a command."""
        if "plan" in result:
            plan_text = "\n".join(result["plan"])
            return self._templates["command"]["acknowledged"].format(
                action=result.get("action", "execute"),
                target=result.get("target", "object"),
                plan=plan_text,
            )
        return self._templates["command"]["cannot_execute"].format(
            action=parsed.get("text", ""),
            reason="no physics engine available"
        )

    def _respond_explanation(self, parsed: dict, result: dict) -> str:
        """Generate response for an explanation request."""
        if "explanation" in result:
            exp = result["explanation"]
            examples = "\n".join(f"  - {ex}" for ex in exp.get("examples", []))
            related = ""
            if result.get("related_objects"):
                related = "Related objects in my experience:\n" + "\n".join(
                    f"  - {obj_id}" for obj_id, _ in result["related_objects"]
                )
            return self._templates["explanation"]["concept"].format(
                concept=result.get("concept", "unknown"),
                definition=exp.get("definition", ""),
                physics=exp.get("physics", ""),
                examples=examples or "  (none yet)",
                related=related,
            )
        return f"I don't have enough knowledge to explain '{parsed.get('text', '')}' yet."

    def _respond_counterfactual(self, parsed: dict, result: dict) -> str:
        """Generate response for a counterfactual question."""
        return self._templates["counterfactual"]["reasoning"].format(
            subject=result.get("subject", "the object"),
            change=result.get("change", "a property"),
            reasoning=result.get("reasoning", "uncertain"),
            prediction=result.get("prediction", "uncertain"),
        )

    def _respond_abstract(self, parsed: dict, result: dict) -> str:
        roots = ", ".join(result.get("root_concepts", [])) or "unknown"
        chains = result.get("chains", [])
        if chains:
            chain_text = "\n".join(
                f"  - {' -> '.join(chain)}" for chain in chains[:6]
            )
        else:
            chain_text = "  - No chain found yet."

        applications = result.get("applications", [])
        if applications:
            app_text = "\n".join(f"  - {item}" for item in applications[:6])
        else:
            app_text = "  - I need more grounded examples before applying this concept."

        return self._templates["abstract"]["reasoning"].format(
            roots=roots,
            chains=chain_text,
            applications=app_text,
        )

    def _respond_social(self, parsed: dict, result: dict) -> str:
        return self._templates["social"]["state"].format(
            agent=result.get("agent", "unknown agent") or "unknown agent",
            answer=result.get("answer", self._summarize_social_state(result)),
        )

    def _respond_memory(self, parsed: dict, result: dict) -> str:
        action = result.get("action")
        records = result.get("records", [])
        if action == "stored" and records:
            return self._templates["memory"]["stored"].format(
                content=records[0].get("content", ""))
        if records:
            lines = "\n".join(
                f"  - [{rec.get('memory_type', 'memory')}] {rec.get('content', '')}"
                for rec in records[:5]
            )
        else:
            lines = "  - No matching memory found yet."
        return self._templates["memory"]["retrieved"].format(records=lines)

    def _respond_transfer(self, parsed: dict, result: dict) -> str:
        hypotheses = result.get("hypotheses", [])
        if hypotheses:
            lines = "\n".join(
                f"  - {h['schema']}: {h['abstraction']} -> "
                f"{', '.join(h['target_hypotheses'])}"
                for h in hypotheses[:5]
            )
        else:
            lines = "  - No reusable schema matched yet; more source-domain experience is needed."
        return self._templates["transfer"]["hypotheses"].format(
            source=result.get("source_domain", "source"),
            target=result.get("target_domain", "target"),
            hypotheses=lines,
        )

    def _respond_abstract_task(self, parsed: dict, result: dict) -> str:
        premises = "\n".join(f"  - {p}" for p in result.get("premises", []))
        rules = "\n".join(f"  - {r}" for r in result.get("rules", []))
        return self._templates["abstract_task"]["result"].format(
            domain=result.get("domain", "abstract").title(),
            premises=premises or "  - None",
            rules=rules or "  - None",
            conclusion=result.get("conclusion", ""),
        )

    def _summarize_social_state(self, result: dict) -> str:
        parts = []
        beliefs = result.get("beliefs", {})
        if beliefs:
            parts.append("Known beliefs: " + "; ".join(beliefs.keys()))
        intentions = result.get("intentions", [])
        if intentions:
            parts.append("Known intentions: " + "; ".join(intentions))
        desires = result.get("desires", [])
        if desires:
            parts.append("Known desires: " + "; ".join(desires))
        emotions = result.get("emotions", {})
        if emotions:
            parts.append("Known emotions: " + "; ".join(emotions.keys()))
        return " ".join(parts) if parts else "No mental-state details are known yet."

    def _respond_unknown(self, parsed: dict) -> str:
        return self._templates["error"]["no_understanding"].format(
            query=parsed.get("text", ""))
