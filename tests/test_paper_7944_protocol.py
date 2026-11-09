"""Checks that the registered implementation settings stay aligned."""
import unittest
import numpy as np
from tempfile import TemporaryDirectory

from credit.paper_7944_protocol import (AttemptBook, CatalogFactory, PAPER_7944,
                                        registered_halving, validate_final_test,
                                        validate_source_groups, validate_task_splits)
from credit.paper_7944_runtime import (FrozenAttemptStore, PaperRetriever,
                                       PhysicalRequest, PhysicalRequestLedger)
from credit.paper_7944_analysis import paired_cluster_interval
from credit.v3.drift import transfer_level
from credit.v3.inference import certify_registered, ppi
from credit.v3.selection import BankSelector, Ledger
from credit.v3.types import Context, Episode, Memory, Node, Role, SideEffect


class PaperProtocolTests(unittest.TestCase):
    @staticmethod
    def registered_manifest():
        splits, groups = {}, {}
        for workload in PAPER_7944.workloads:
            offset, parts, mapping = 0, {}, {}
            for part, count in (("source", workload.source_task_ids),
                                ("development", workload.development_task_ids),
                                ("certification", workload.certification_task_ids),
                                ("test", workload.test_task_ids)):
                parts[part] = [f"{workload.name}-{i}" for i in range(offset, offset + count)]
                for i, task in enumerate(parts[part]):
                    count_groups = workload.test_source_groups if part == "test" else count
                    mapping[task] = f"{part}-group-{i % count_groups}"
                offset += count
            splits[workload.name], groups[workload.name] = parts, mapping
        return splits, groups

    def test_registered_search_and_workloads(self):
        manifest = PAPER_7944.manifest()
        self.assertEqual(manifest["new_search_pairs_total"], 1536)
        self.assertEqual([w["test_task_ids"] for w in manifest["workloads"]], [80, 70, 25])
        self.assertEqual([w["prediction_draws_per_guard"] for w in manifest["workloads"]],
                         [2000, 4000, 3000])
        splits = {}
        for workload in PAPER_7944.workloads:
            offset, parts = 0, {}
            for part, count in (("source", workload.source_task_ids),
                                ("development", workload.development_task_ids),
                                ("certification", workload.certification_task_ids),
                                ("test", workload.test_task_ids)):
                parts[part] = list(range(offset, offset + count))
                offset += count
            splits[workload.name] = parts
        validate_task_splits(splits)
        splits["Retail"]["test"].pop()
        with self.assertRaises(ValueError):
            validate_task_splits(splits)

    def test_source_group_and_final_test_population(self):
        splits, groups = self.registered_manifest()
        validate_source_groups(splits, groups)
        role = Role(("id",), ("id",), SideEffect.RO, "read")
        tasks = splits["ToolSandbox"]["test"]
        contexts = tuple(Context(f"episode-{i}", "guard", (Node("n", role, "v"),),
                                 (), i, task_id=tasks[i % len(tasks)],
                                 source_group=groups["ToolSandbox"][tasks[i % len(tasks)]])
                         for i in range(2000))
        kinds = {task: "new_workflow" if i < 44 else "changed_contract"
                 for i, task in enumerate(tasks)}
        validate_final_test("ToolSandbox", contexts, splits, groups, kinds)
        corrupted = dict(groups["ToolSandbox"])
        corrupted[tasks[0]] = "source-group-0"
        groups["ToolSandbox"] = corrupted
        with self.assertRaises(ValueError):
            validate_final_test("ToolSandbox", contexts, splits, groups, kinds)

    def test_paper_cluster_interval_resamples_groups(self):
        baseline = np.zeros(2000, dtype=int)
        candidate = np.zeros(2000, dtype=int)
        candidate[:200] = 1
        groups = [f"group-{i % 30}" for i in range(2000)]
        result = paired_cluster_interval(baseline, candidate, groups,
                                         workload="ToolSandbox", seed=42)
        self.assertAlmostEqual(result["paired_difference"], 0.1)
        self.assertEqual(result["bootstrap_resamples"], 10000)
        self.assertEqual(result["source_groups"], 30)
        self.assertLessEqual(result["lower_95"], 0.1)
        self.assertGreaterEqual(result["upper_95"], 0.1)

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
        selector.models = (type("FrozenEnsemble", (), {
            "paper_7944_ensemble": True, "fit_report": {"frozen": True}})(),)
        selector.retrieve = PaperRetriever((Memory("memory", "hint", (1,), None, "unit"),),
                                           {"memory": (1.0, 0.0)}, verify_tokens=False)
        selector.retrieve.verify_tokens = True  # The mocked selector does not perform ranking.
        selector.ledger = Ledger({"certification": 2400}, request_ledger=PhysicalRequestLedger(),
                                 require_request_ids=True)
        selector.transported_strata = frozenset(selector.mixture)
        phases, frozen = [], []
        role = Role(("id",), ("id",), SideEffect.RO, "read")
        selector.contexts = lambda phase, count: (phases.append((phase, count)) or
            {guard: tuple(Context(f"{phase}-{guard}-{i}", guard,
                                  (Node("n", role, "v"),), (), i,
                                  initial_user_text="cached", initial_embedding=(1.0, 0.0))
                          for i in range(count)) for guard in selector.mixture})
        selector.predictions = lambda contexts, bank: np.zeros(len(contexts))
        selector._executions = lambda phase, contexts, banks: {
            bank: {guard: np.zeros(len(rows)) for guard, rows in contexts.items()}
            for bank in banks}
        selector.memories = (Memory("memory", "hint", (1,), None, "unit"),)
        result = selector.switch_registered((), ("memory",), workload="ToolSandbox",
                                            attempt_index=2,
                                            residual_weights={guard: 0.2 for guard in selector.mixture},
                                            freeze=frozen.append)
        self.assertEqual(phases, [("certification", 200),
                                  ("certification-unlabeled", 2000)])
        self.assertEqual(result["certificates"][0]["estimates"]["guard-0"]["weight"], 0.2)
        self.assertEqual(frozen[0]["attempt_index"], 2)
        self.assertEqual(frozen[0]["finalists"], [("memory",)])
        self.assertTrue(result["registered_single_finalist"])
        with self.assertRaises(ValueError):
            selector.switch_registered((), ("memory",), workload="ToolSandbox",
                                       attempt_index=3,
                                       residual_weights={guard: 0.2 for guard in selector.mixture},
                                       freeze=None)

    def test_catalog_sampling_is_fresh_and_with_replacement(self):
        role = Role(("id",), ("id",), SideEffect.RO, "read")
        catalog = tuple(Context(f"catalog-{i}", "guard", (Node("n", role, "v"),),
                                (), i, initial_state=i, initial_user_text=f"message-{i}",
                                initial_embedding=(1.0, 0.0)) for i in range(256))
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
                             workload="ToolSandbox", residual_weights={},
                             freeze=lambda _: None)
        second = book.attempt(BudgetAbandoned(), (), ("candidate",),
                              workload="ToolSandbox", residual_weights={},
                              freeze=lambda _: None)
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

    def test_paper_retrieval_uses_frozen_embedding_and_role_scope(self):
        role = Role(("id",), ("id",), SideEffect.RO, "read")
        other = Role(("id",), ("id",), SideEffect.IR, "write")
        memories = (Memory("a", "a", (1,), frozenset((role.key,)), "unit"),
                    Memory("b", "b", (2,), frozenset((role.key,)), "unit"),
                    Memory("c", "c", (3,), frozenset((other.key,)), "unit"))
        retriever = PaperRetriever(memories, {"a": (1.0, 0.0),
                                              "b": (0.8, 0.6), "c": (1.0, 0.0)},
                                   verify_tokens=False)
        context = Context("task", "guard", (Node("n", role, "v"),), (), 1,
                          initial_user_text="cached", initial_embedding=(1.0, 0.0))
        self.assertEqual(retriever(context, ("a", "b", "c")), ("a", "b"))
        with self.assertRaises(ValueError):
            retriever(Context("missing", "guard", (Node("n", role, "v"),), (), 2),
                      ("a",))

    def test_physical_ledger_counts_failures_once(self):
        requests = PhysicalRequestLedger()
        requests.charge_many((PhysicalRequest("one", "catalog", "shared", "initial",
                                              "simulator", "ok", 10, 5,
                                              model=PAPER_7944.simulator_model,
                                              temperature=0.7, top_p=1.0),
                              PhysicalRequest("two", "catalog", "shared", "initial",
                                              "embedding", "failed", 8, 0,
                                              model=PAPER_7944.embedding_model)))
        self.assertEqual(requests.report()["physical_requests"], 2)
        self.assertEqual(requests.report()["status_counts"]["failed"], 1)
        with self.assertRaises(ValueError):
            requests.charge(PhysicalRequest("one", "certification", "pair", "candidate",
                                            "actor", "ok"))
        ledger = Ledger({"certification": 1}, request_ledger=requests,
                        require_request_ids=True)
        role = Role(("id",), ("id",), SideEffect.RO, "read")
        context = Context("task", "guard", (Node("n", role, "v"),), (), 1)
        episode = Episode(context, (), (), True, 2, "a", "b", "b", True,
                          request_ids=("one", "two"))
        ledger.charge("certification", episode)
        self.assertEqual(ledger.calls["certification"], 2)
        with TemporaryDirectory() as directory:
            requests.save(f"{directory}/requests.jsonl")
            store = FrozenAttemptStore(directory)
            store({"workload": "ToolSandbox", "attempt_index": 1,
                   "incumbent": [], "finalists": [["memory"]]})
            with self.assertRaises(FileExistsError):
                store({"workload": "ToolSandbox", "attempt_index": 1,
                       "incumbent": [], "finalists": [["memory"]]})

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
