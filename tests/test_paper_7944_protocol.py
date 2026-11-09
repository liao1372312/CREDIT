"""Checks that the registered implementation settings stay aligned."""
import unittest

from credit.paper_7944_protocol import (AttemptBook, CatalogFactory, PAPER_7944,
                                        registered_halving)
from credit.v3.drift import transfer_level
from credit.v3.inference import certify_registered, ppi
from credit.v3.selection import BankSelector
from credit.v3.types import Context, Node, Role, SideEffect


class PaperProtocolTests(unittest.TestCase):
    def test_registered_search_and_workloads(self):
        manifest = PAPER_7944.manifest()
        self.assertEqual(manifest["new_search_pairs_total"], 1536)
        self.assertEqual([w["test_task_ids"] for w in manifest["workloads"]], [80, 70, 25])
        self.assertEqual([w["prediction_draws_per_guard"] for w in manifest["workloads"]],
                         [2000, 4000, 3000])

    def test_lifetime_error_spending_and_guard(self):
        estimates = {"protected": ppi([0.2] * 200, [0.0] * 200, [0.0] * 2000)}
        first = certify_registered(estimates, {"protected": 1.0}, attempt_index=1)
        second = certify_registered(estimates, {"protected": 1.0}, attempt_index=2)
        self.assertAlmostEqual(first["attempt_alpha"], 0.025)
        self.assertAlmostEqual(second["attempt_alpha"], 0.05 / 6)
        self.assertEqual(first["candidate_count"], 1)
        self.assertEqual(first["familywise_error_budget"], first["attempt_alpha"])
        with self.assertRaises(ValueError):
            certify_registered(estimates, {"protected": 1.0}, attempt_index=0)

    def test_paper_drift_thresholds(self):
        role = Role(("id",), ("id",), SideEffect.ID, "read")
        low_support = transfer_level(role, role, "target", changed=0, fixtures=80, support=29)
        verified = transfer_level(role, role, "target", changed=0, fixtures=80, support=30)
        self.assertEqual(low_support.level.value, "X")
        self.assertEqual(verified.level.value, "V")

    def test_registered_selector_freezes_one_finalist_and_spends_alpha(self):
        selector = BankSelector.__new__(BankSelector)
        selector.mixture = {f"guard-{i}": 1 / 6 for i in range(6)}
        selector.capacity = 800
        selector.radius = 0.25
        selector.cost_penalty = 1.25e-5
        selector.memories = ()
        selector.factory = CatalogFactory.__new__(CatalogFactory)
        selector.factory.attempt_index = 2
        selector.factory.catalogs = {guard: () for guard in selector.mixture}
        calls, frozen = [], []

        def simulated_switch(incumbent, finalists, **kwargs):
            calls.append((incumbent, finalists, kwargs))
            kwargs["freeze"]({"incumbent": incumbent, "finalists": finalists})
            return {"bank": incumbent, "switched": False}

        selector.switch = simulated_switch
        from credit.v3.types import Memory
        selector.memories = (Memory("memory", "hint", (1,), None, "unit"),)
        result = selector.switch_registered((), ("memory",), workload="ToolSandbox",
                                            attempt_index=2, freeze=frozen.append)
        self.assertEqual(calls[0][1], [("memory",)])
        self.assertEqual(calls[0][2]["certification_n"], 200)
        self.assertEqual(calls[0][2]["unlabeled_n"], 2000)
        self.assertAlmostEqual(calls[0][2]["alpha1"], 0.05 / 12)
        self.assertEqual(frozen[0]["attempt_index"], 2)
        self.assertTrue(result["registered_single_finalist"])
        with self.assertRaises(ValueError):
            selector.switch_registered((), ("memory",), workload="ToolSandbox",
                                       attempt_index=3, freeze=None)

    def test_catalog_sampling_is_fresh_and_with_replacement(self):
        role = Role(("id",), ("id",), SideEffect.RO, "read")
        catalog = tuple(Context(f"catalog-{i}", "guard", (Node("n", role, "v"),),
                                (), i, initial_state=i) for i in range(256))
        factory = CatalogFactory({"guard": catalog}, seed=31, attempt_index=1)
        labeled = [factory("guard", "certification", i) for i in range(400)]
        predictions = [factory("guard", "certification-unlabeled", i) for i in range(400)]
        self.assertEqual(len({row.uid for row in labeled}), 400)
        self.assertEqual(len({row.seed for row in labeled}), 400)
        self.assertLess(len({row.initial_state for row in labeled}), 400)
        self.assertNotEqual([row.initial_state for row in labeled],
                            [row.initial_state for row in predictions])
        self.assertEqual(labeled, [factory("guard", "certification", i) for i in range(400)])
        next_attempt = CatalogFactory({"guard": catalog}, seed=31, attempt_index=2)
        self.assertNotEqual([row.uid for row in labeled],
                            [next_attempt("guard", "certification", i).uid for i in range(400)])

    def test_abandoned_attempt_spends_lifetime_index(self):
        book = AttemptBook()

        class BudgetAbandoned:
            def switch_registered(self, *args, **kwargs):
                return {"switched": False, "reason": "insufficient certification budget",
                        "attempt_index": kwargs["attempt_index"]}

        first = book.attempt(BudgetAbandoned(), (), ("candidate",),
                             workload="ToolSandbox", freeze=lambda _: None)
        second = book.attempt(BudgetAbandoned(), (), ("candidate",),
                              workload="ToolSandbox", freeze=lambda _: None)
        self.assertEqual((first["attempt_index"], second["attempt_index"]), (1, 2))

    def test_registered_halving_pools_all_six_stages(self):
        banks = [(f"bank-{i}",) for i in range(32)]
        acquired = []

        def acquire(bank, n, stage):
            acquired.append((bank, n, stage))
            return [(bank, stage, i) for i in range(n)]

        result = registered_halving(banks, acquire,
                                    lambda bank, history: -int(bank[0].split("-")[1]))
        self.assertEqual(result["finalist"], ("bank-0",))
        self.assertEqual(result["development_pair_count"], 1536)
        self.assertEqual(len(result["finalist_history"]), sum(PAPER_7944.new_pairs_per_bank))
        self.assertEqual([row["pair_count"] for row in result["stages"]], [256] * 6)
        self.assertEqual(len(acquired), sum(PAPER_7944.candidate_counts))

    def test_paper_model_architecture_when_neural_dependencies_exist(self):
        try:
            import torch
            from credit.paper_7944_model import PaperHeads
        except ImportError:
            self.skipTest("optional neural dependencies are unavailable")
        heads = PaperHeads(role_types=2)
        for component in (heads.baseline, heads.unary, heads.pair):
            self.assertEqual([layer.out_features for layer in component
                              if isinstance(layer, torch.nn.Linear)], [128, 128, 2])
            self.assertEqual([layer.p for layer in component
                              if isinstance(layer, torch.nn.Dropout)], [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
