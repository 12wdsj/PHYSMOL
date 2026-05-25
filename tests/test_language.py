"""PHYSMOL Language Cognitive Layer Tests.

Tests cover:
  1. VSA recipe store (attribute primitives, recipes, resonance)
  2. Text encoder (tokenization, word vectors, sentence encoding)
  3. Semantic parser (intent classification, attribute extraction, object matching)
  4. Reasoning engine (prediction, counterfactual, explanation, action planning)
  5. Cognitive interface (end-to-end query processing)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))


# ---------------------------------------------------------------------------
# VSA Store Tests
# ---------------------------------------------------------------------------

class TestAttributePrimitivePool:
    def test_create(self):
        from physmol.vsa_store import AttributePrimitivePool
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        assert pool.vsa_dim == 1000
        assert len(pool) > 0

    def test_default_categories(self):
        from physmol.vsa_store import AttributePrimitivePool
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        cats = pool.list_categories()
        assert "shape" in cats
        assert "color" in cats
        assert "material" in cats
        assert "elasticity" in cats

    def test_get_primitive(self):
        from physmol.vsa_store import AttributePrimitivePool
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        vec = pool.get("shape", "sphere")
        assert vec is not None
        assert len(vec) == 1000
        assert set(np.unique(vec)).issubset({-1.0, 1.0})

    def test_category_tag(self):
        from physmol.vsa_store import AttributePrimitivePool
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        tag = pool.get_tag("color")
        assert tag is not None
        # Different categories should have different tags
        tag2 = pool.get_tag("shape")
        assert not np.array_equal(tag, tag2)

    def test_resolve_id(self):
        from physmol.vsa_store import AttributePrimitivePool
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        cat, name = pool.resolve_id("shape_sphere")
        assert cat == "shape"
        assert name == "sphere"

    def test_get_by_id(self):
        from physmol.vsa_store import AttributePrimitivePool
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        vec = pool.get_by_id("color_red")
        assert vec is not None
        # Should be same as get("color", "red")
        vec2 = pool.get("color", "red")
        assert np.array_equal(vec, vec2)


class TestRecipeStore:
    def test_register_and_retrieve(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)

        store.register_recipe("ball_1", ["shape_sphere", "color_red", "material_rubber"])
        assert "ball_1" in store
        assert store.get_recipe("ball_1") == ["shape_sphere", "color_red", "material_rubber"]

    def test_synthesize(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)

        store.register_recipe("ball_1", ["shape_sphere", "color_red"])
        vec = store.synthesize("ball_1")
        assert vec is not None
        assert len(vec) == 1000
        assert np.isfinite(vec).all()
        # Should be normalized
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_synthesize_nonexistent(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)
        assert store.synthesize("nonexistent") is None

    def test_resonate(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)

        store.register_recipe("ball_1", ["shape_sphere", "color_red"])
        store.register_recipe("cube_1", ["shape_cube", "color_blue"])

        # Query with a red sphere vector should match ball_1 best
        red_sphere = store.synthesize_from_ids(["shape_sphere", "color_red"])
        matches = store.resonate(red_sphere, top_k=2)
        assert len(matches) == 2
        assert matches[0][0] == "ball_1"
        assert matches[0][1] > matches[1][1]  # higher similarity

    def test_decompose(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)

        # Create a known vector and decompose it
        red_sphere = store.synthesize_from_ids(["shape_sphere", "color_red"])
        decomp = store.decompose(red_sphere)

        assert "shape" in decomp
        assert "color" in decomp
        assert decomp["shape"][0] == "sphere"
        assert decomp["color"][0] == "red"

    def test_find_by_attributes(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)

        store.register_recipe("ball_1", ["shape_sphere", "color_red", "material_rubber"])
        store.register_recipe("cube_1", ["shape_cube", "color_blue", "material_metal"])
        store.register_recipe("ball_2", ["shape_sphere", "color_blue"])

        # Search for spheres
        results = store.find_by_attributes(["shape_sphere"])
        assert len(results) == 2
        obj_ids = [r[0] for r in results]
        assert "ball_1" in obj_ids
        assert "ball_2" in obj_ids

    def test_remove_recipe(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)

        store.register_recipe("ball_1", ["shape_sphere"])
        assert "ball_1" in store
        store.remove_recipe("ball_1")
        assert "ball_1" not in store


class TestConceptSynthesizer:
    def test_compose_concept(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore, ConceptSynthesizer
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)
        synth = ConceptSynthesizer(store)

        vec = synth.compose_concept({"shape": "sphere", "color": "red"})
        assert len(vec) == 1000
        assert np.linalg.norm(vec) > 0

    def test_explain_concept(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore, ConceptSynthesizer
        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)
        store.register_recipe("ball_1", ["shape_sphere", "color_red", "material_rubber"])
        synth = ConceptSynthesizer(store)

        explanation = synth.explain_concept("ball_1")
        assert explanation["shape"] == "sphere"
        assert explanation["color"] == "red"
        assert explanation["material"] == "rubber"


# ---------------------------------------------------------------------------
# Text Encoder Tests
# ---------------------------------------------------------------------------

class TestTextToVSA:
    def test_create(self):
        from physmol.language.text_encoder import TextToVSA
        enc = TextToVSA(vsa_dim=1000, seed=42)
        assert enc.vsa_dim == 1000

    def test_tokenize(self):
        from physmol.language.text_encoder import TextToVSA
        enc = TextToVSA(vsa_dim=1000, seed=42)
        tokens = enc.tokenize("What happens if I drop the red ball?")
        assert "what" in tokens
        assert "happens" in tokens
        assert "drop" in tokens
        assert "red" in tokens
        assert "ball" in tokens

    def test_encode_word(self):
        from physmol.language.text_encoder import TextToVSA
        enc = TextToVSA(vsa_dim=1000, seed=42)
        v1 = enc.encode_word("ball")
        v2 = enc.encode_word("ball")
        assert np.array_equal(v1, v2)  # same word -> same vector
        v3 = enc.encode_word("cube")
        assert not np.array_equal(v1, v3)  # different word -> different vector

    def test_encode_sentence(self):
        from physmol.language.text_encoder import TextToVSA
        enc = TextToVSA(vsa_dim=1000, seed=42)
        v = enc.encode("the red ball")
        assert len(v) == 1000
        assert np.isfinite(v).all()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_different_sentences_different_vectors(self):
        from physmol.language.text_encoder import TextToVSA
        enc = TextToVSA(vsa_dim=1000, seed=42)
        v1 = enc.encode("the red ball")
        v2 = enc.encode("the blue cube")
        # In high dimensions, these should be nearly orthogonal
        sim = np.dot(v1, v2)
        assert abs(sim) < 0.5  # not identical

    def test_physics_vocabulary(self):
        from physmol.language.text_encoder import TextToVSA
        enc = TextToVSA(vsa_dim=1000, seed=42)
        # These words should be pre-populated
        assert enc.lexicon.has_word("ball")
        assert enc.lexicon.has_word("fall")
        assert enc.lexicon.has_word("gravity")
        assert enc.lexicon.has_word("elastic")


# ---------------------------------------------------------------------------
# Semantic Parser Tests
# ---------------------------------------------------------------------------

class TestSemanticParser:
    def _make_parser(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        from physmol.language.text_encoder import TextToVSA
        from physmol.language.semantic_parser import SemanticParser

        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)
        enc = TextToVSA(vsa_dim=1000, seed=42)
        return SemanticParser(enc, store), store

    def test_classify_intent_question(self):
        parser, _ = self._make_parser()
        intent, conf = parser.classify_intent("What happens if I drop the ball?")
        assert intent == "question"
        assert conf > 0

    def test_classify_intent_command(self):
        parser, _ = self._make_parser()
        intent, conf = parser.classify_intent("Push the block to the top")
        assert intent == "command"
        assert conf > 0

    def test_classify_intent_explanation(self):
        parser, _ = self._make_parser()
        intent, conf = parser.classify_intent("Explain elasticity")
        assert intent == "explanation"
        assert conf > 0

    def test_classify_intent_counterfactual(self):
        parser, _ = self._make_parser()
        intent, conf = parser.classify_intent("What if the ball was heavier?")
        assert intent == "counterfactual"
        assert conf > 0

    def test_extract_attribute_hints(self):
        parser, _ = self._make_parser()
        hints = parser.extract_attribute_hints(["red", "ball", "rubber"])
        assert "color_red" in hints
        assert "shape_sphere" in hints  # "ball" -> sphere
        assert "material_rubber" in hints

    def test_find_matching_objects(self):
        parser, store = self._make_parser()
        store.register_recipe("ball_1", ["shape_sphere", "color_red"])
        store.register_recipe("cube_1", ["shape_cube", "color_blue"])

        matches = parser.find_matching_objects("the red ball")
        assert len(matches) > 0
        assert matches[0][0] == "ball_1"

    def test_parse_query(self):
        parser, store = self._make_parser()
        store.register_recipe("ball_1", ["shape_sphere", "color_red"])
        result = parser.parse_query("What happens if I drop the red ball?")

        assert result["intent"] == "question"
        assert "drop" in result["tokens"]
        assert "red" in result["tokens"]
        assert "ball" in result["tokens"]
        assert len(result["matching_objects"]) > 0


# ---------------------------------------------------------------------------
# Reasoning Engine Tests
# ---------------------------------------------------------------------------

class TestReasoningEngine:
    def _make_engine(self):
        from physmol.vsa_store import AttributePrimitivePool, RecipeStore
        from physmol.language.reasoning import ReasoningEngine

        pool = AttributePrimitivePool(vsa_dim=1000, seed=42)
        store = RecipeStore(pool)
        store.register_recipe("ball_1", ["shape_sphere", "color_red", "material_rubber", "elasticity_elastic"])
        store.register_recipe("cube_1", ["shape_cube", "color_blue", "material_metal", "elasticity_rigid"])
        return ReasoningEngine(store)

    def test_predict_drop(self):
        engine = self._make_engine()
        result = engine.predict("ball", {"action": "drop"})
        assert "prediction" in result
        assert len(result["prediction"]) > 0

    def test_predict_collide(self):
        engine = self._make_engine()
        result = engine.predict("ball", {"action": "collide"})
        assert "prediction" in result

    def test_counterfactual_mass(self):
        engine = self._make_engine()
        result = engine.counterfactual("ball", "mass (heavier)")
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0

    def test_counterfactual_gravity(self):
        engine = self._make_engine()
        result = engine.counterfactual("ball", "gravity (stronger)")
        assert "reasoning" in result

    def test_explain_elasticity(self):
        engine = self._make_engine()
        result = engine.explain_concept("elasticity")
        assert result["concept"] == "elasticity"
        assert "definition" in result["explanation"]
        assert "physics" in result["explanation"]
        assert len(result["explanation"]["examples"]) > 0

    def test_explain_gravity(self):
        engine = self._make_engine()
        result = engine.explain_concept("gravity")
        assert result["concept"] == "gravity"
        assert "9.8" in result["explanation"]["physics"]

    def test_explain_momentum(self):
        engine = self._make_engine()
        result = engine.explain_concept("momentum")
        assert "p = mv" in result["explanation"]["physics"]

    def test_plan_action_push(self):
        engine = self._make_engine()
        result = engine.plan_action("push the block to the top")
        assert result["action"] == "push"
        assert result["target"] == "block"
        assert len(result["plan"]) > 0

    def test_plan_action_drop(self):
        engine = self._make_engine()
        result = engine.plan_action("drop the ball")
        assert result["action"] == "drop"
        assert result["target"] == "ball"


# ---------------------------------------------------------------------------
# Cognitive Interface Tests
# ---------------------------------------------------------------------------

class TestCognitiveInterface:
    def _make_ci(self):
        from physmol.language.cognitive import CognitiveInterface
        ci = CognitiveInterface(vsa_dim=1000, seed=42)
        ci.register_object("ball_1", ["shape_sphere", "color_red", "material_rubber", "elasticity_elastic"])
        ci.register_object("cube_1", ["shape_cube", "color_blue", "material_metal", "elasticity_rigid"])
        return ci

    def test_create(self):
        ci = self._make_ci()
        status = ci.get_status()
        assert status["num_objects"] == 2
        assert status["vsa_dim"] == 1000

    def test_register_object(self):
        ci = self._make_ci()
        ci.register_object("cyl_1", ["shape_cylinder", "color_green"])
        assert "cyl_1" in ci.list_objects()

    def test_get_object_attributes(self):
        ci = self._make_ci()
        attrs = ci.get_object_attributes("ball_1")
        assert attrs["shape"] == "sphere"
        assert attrs["color"] == "red"
        assert attrs["material"] == "rubber"

    def test_query_question(self):
        ci = self._make_ci()
        response = ci.query("What happens if I drop the red ball?")
        assert len(response) > 0
        assert isinstance(response, str)

    def test_query_explanation(self):
        ci = self._make_ci()
        response = ci.query("Explain elasticity")
        assert "elasticity" in response.lower() or "elastic" in response.lower()

    def test_query_command(self):
        ci = self._make_ci()
        response = ci.query("Push the cube to the top")
        assert len(response) > 0
        assert "push" in response.lower() or "cube" in response.lower()

    def test_query_counterfactual(self):
        ci = self._make_ci()
        response = ci.query("What if the ball was heavier?")
        assert len(response) > 0

    def test_conversation_history(self):
        ci = self._make_ci()
        ci.query("What is gravity?")
        ci.query("Explain momentum")
        history = ci.get_history()
        assert len(history) == 2
        assert history[0]["user"] == "What is gravity?"
        assert history[1]["user"] == "Explain momentum"

    def test_multiple_queries(self):
        ci = self._make_ci()
        queries = [
            "What happens if I drop the ball?",
            "Explain elasticity",
            "Push the block left",
            "What if gravity was stronger?",
            "What is friction?",
        ]
        for q in queries:
            response = ci.query(q)
            assert len(response) > 0
            assert isinstance(response, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
