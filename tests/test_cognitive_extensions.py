"""Tests for abstract reasoning, intrinsic motivation, and theory of mind."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))


def test_abstract_reasoning_justice_to_law_and_punishment():
    from physmol.language.abstract_reasoning import AbstractConceptReasoner

    reasoner = AbstractConceptReasoner()
    result = reasoner.infer("If we know justice, what follows for law and crime?")

    assert "justice" in result.activated_concepts
    assert "law" in result.activated_concepts
    assert "punishment" in result.activated_concepts
    assert any("proportionate" in item for item in result.applications)


def test_abstract_reasoning_democracy_to_slavery_wrongness():
    from physmol.language.abstract_reasoning import AbstractConceptReasoner

    reasoner = AbstractConceptReasoner()
    result = reasoner.infer("民主意味着自由和平等，那么奴隶制为什么是错误的？")

    assert "democracy" in result.activated_concepts
    assert "freedom" in result.activated_concepts
    assert "slavery" in result.activated_concepts
    assert "wrongness" in result.activated_concepts


def test_intrinsic_motivation_tracks_learning_progress():
    from physmol.motivation import IntrinsicMotivationSystem

    motivation = IntrinsicMotivationSystem()
    motivation.record_outcome("drop:red_ball", 0.8, uncertainty=0.2)
    motivation.record_outcome("drop:red_ball", 0.6, uncertainty=0.2)
    result = motivation.record_outcome("drop:red_ball", 0.2, uncertainty=0.1)

    assert result["intrinsic_reward"] > 0
    assert "learning_progress" in result


def test_theory_of_mind_updates_and_answers_beliefs():
    from physmol.language.theory_of_mind import TheoryOfMindModel

    tom = TheoryOfMindModel()
    update = tom.update_from_text("Alice believes that the ball is in the box")
    answer = tom.answer("What does Alice believe?")

    assert update is not None
    assert update["agent"] == "Alice"
    assert "ball is in the box" in answer["answer"]


def test_cognitive_interface_handles_new_cognition_paths():
    from physmol.language.cognitive import CognitiveInterface

    ci = CognitiveInterface(vsa_dim=512, seed=42)
    abstract = ci.query("Explain why democracy conflicts with slavery.")
    ci.query("Alice believes that the cube is behind the wall")
    social = ci.query("What does Alice believe?")
    hello = ci.chat("hello")

    assert "slavery" in abstract.lower() or "freedom" in abstract.lower()
    assert "cube is behind the wall" in social
    assert "Hello" in hello


def test_world_model_reports_intrinsic_reward():
    from physmol.world_model import HierarchicalWorldModel

    model = HierarchicalWorldModel(state_dim=3)
    state = np.zeros(3, dtype=np.float32)
    observed = np.ones(3, dtype=np.float32)
    result = model.step(state, observed_next=observed)

    assert "intrinsic_reward" in result
    assert result["intrinsic_reward"] > 0


def test_long_term_memory_stores_and_retrieves_facts():
    from physmol.language.text_encoder import TextToVSA
    from physmol.long_term_memory import LongTermMemory

    memory = LongTermMemory(TextToVSA(vsa_dim=256))
    memory.add_fact("rubber ball", "bounces on", "hard floor", tags=["physics"])
    records = memory.retrieve("what bounces on hard floor", top_k=1)

    assert records
    assert "rubber ball" in records[0][0].content


def test_transfer_blocks_to_chess_produces_hypotheses():
    from physmol.transfer import CrossDomainTransferEngine

    transfer = CrossDomainTransferEngine()
    patterns = transfer.lift_experience_to_patterns(
        "A block can obstruct a path and a later move changes the state")
    result = transfer.propose_transfer("blocks", "chess", patterns)

    assert result["hypotheses"]
    assert any("blocking_constraint" in h["target_hypotheses"]
               for h in result["hypotheses"])


def test_abstract_task_reasoner_math_proof():
    from physmol.language.abstract_tasks import AbstractTaskReasoner

    result = AbstractTaskReasoner().reason("prove even + even is even")

    assert result["domain"] == "math"
    assert "sum of two even integers is even" in result["conclusion"]


def test_training_data_normalizes_instruction_rows():
    from physmol.training_data import normalize_dataset_row

    row = {"instruction": "Explain gravity", "output": "Gravity attracts masses."}
    ex = normalize_dataset_row(row, source="unit")

    assert ex.text == "Explain gravity"
    assert ex.target == "Gravity attracts masses."


def test_cognitive_interface_memory_transfer_and_abstract_tasks():
    from physmol.language.cognitive import CognitiveInterface

    ci = CognitiveInterface(vsa_dim=512)
    stored = ci.query("remember that rubber balls bounce on hard floors")
    recalled = ci.query("what do you remember about rubber balls?")
    transfer = ci.query("How can knowledge from playing with blocks transfer to chess?")
    proof = ci.query("prove even + even is even")

    assert "stored" in stored.lower()
    assert "rubber balls" in recalled.lower()
    assert "cross-domain transfer" in transfer.lower()
    assert "sum of two even integers is even" in proof.lower()


def test_abstract_cognition_trainer_learns_components(tmp_path):
    from physmol.abstract_training import AbstractCognitionTrainer
    from physmol.progress import ProgressLogger
    from physmol.training_data import TrainingExample

    examples = [
        TrainingExample("abstract", "prove even + even is even", "2a + 2b = 2(a+b)"),
        TrainingExample("abstract", "Legal case: crime evidence intent punishment", "Use due process and proportionate punishment."),
        TrainingExample("abstract", "Slavery is wrong because it denies freedom.", "Avoid domination because it violates freedom."),
    ]
    out_dir = str(tmp_path / "abstract")
    trainer = AbstractCognitionTrainer()
    metrics = trainer.train(examples, progress=ProgressLogger(out_dir))
    trainer.save(out_dir)

    assert metrics["proof_rules"] >= 1
    assert metrics["legal_cases"] >= 1
    assert metrics["value_constraints"] >= 1
