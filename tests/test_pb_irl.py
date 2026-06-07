import unittest
import numpy as np
import torch


class TestMultiRewardAgg(unittest.TestCase):
    def test_mean_is_sum_over_n(self):
        from trainer.irl_trainer import MultiRewardNetwork
        torch.manual_seed(0)
        N, D = 8, 6
        net_sum = MultiRewardNetwork(input_dim=D, num_stocks=N, agg="sum")
        net_mean = MultiRewardNetwork(input_dim=D, num_stocks=N, agg="mean")
        net_mean.load_state_dict(net_sum.state_dict())  # identical params
        state = torch.randn(N, D)
        action = (torch.rand(N) > 0.5).float()
        with torch.no_grad():
            r_sum = net_sum(state, action)
            r_mean = net_mean(state, action)
        # guard against a degenerate all-zero output passing trivially (0 == 0*N)
        self.assertFalse(torch.allclose(r_sum, torch.zeros_like(r_sum)))
        self.assertTrue(torch.allclose(r_sum, r_mean * N, atol=1e-5))


class _DummyNet(torch.nn.Module):
    """Reward net stub: output = sum of the state tensor (ignores action)."""
    def forward(self, state, action):
        return state.sum().reshape(1)


class TestDecodeVariant(unittest.TestCase):
    def test_table(self):
        from trainer.pb_irl import decode_variant
        self.assertEqual(decode_variant("base"), ("sum", False, True))
        self.assertEqual(decode_variant("mean-reward"), ("mean", False, True))
        self.assertEqual(decode_variant("norm"), ("sum", True, True))
        self.assertEqual(decode_variant("norm-nokl"), ("sum", True, False))

    def test_unknown_raises(self):
        from trainer.pb_irl import decode_variant
        with self.assertRaises(ValueError):
            decode_variant("nope")


class TestRewardNormalizer(unittest.TestCase):
    def test_forward_applies_zscore(self):
        from trainer.pb_irl import RewardNormalizer
        rn = RewardNormalizer(_DummyNet())
        rn.mu.copy_(torch.tensor([2.0]))
        rn.sigma.copy_(torch.tensor([4.0]))
        s = torch.ones(3, 2)  # sum = 6
        out = rn(s, torch.zeros(3))
        self.assertAlmostEqual(out.item(), (6.0 - 2.0) / 4.0, places=5)

    def test_update_sets_mu_sigma(self):
        from trainer.pb_irl import RewardNormalizer
        rn = RewardNormalizer(_DummyNet())
        trajs = [(np.ones((1, 2), dtype=np.float32), np.zeros(1, dtype=np.float32)),   # sum 2
                 (np.ones((1, 4), dtype=np.float32), np.zeros(1, dtype=np.float32))]   # sum 4
        rn.update(trajs, device="cpu")
        self.assertAlmostEqual(rn.mu.item(), 3.0, places=5)
        expected_sigma = torch.tensor([2.0, 4.0]).std().item()
        self.assertAlmostEqual(rn.sigma.item(), expected_sigma, places=5)


class TestPrefHelpers(unittest.TestCase):
    def test_forward_sharpe_zero_when_constant(self):
        from trainer.pb_irl import portfolio_forward_sharpe
        fl = [np.array([0.1, 0.1, 0.0])] * 3   # basket mean constant -> std 0
        self.assertEqual(portfolio_forward_sharpe(np.array([0, 1]), fl), 0.0)

    def test_forward_sharpe_value(self):
        from trainer.pb_irl import portfolio_forward_sharpe
        fl = [np.array([0.2, 0.0]), np.array([0.1, 0.0]), np.array([0.0, 0.0])]
        daily = np.array([0.1, 0.05, 0.0])     # mean over selected=[0,1] per day
        self.assertAlmostEqual(
            portfolio_forward_sharpe(np.array([0, 1]), fl),
            daily.mean() / daily.std(), places=6)

    def test_build_pairs_keeps_and_weights(self):
        from trainer.pb_irl import build_preference_pairs
        pairs = build_preference_pairs([1.0, 0.0], [0, 1], margin=0.5, recency=0.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0][0], pairs[0][1]), (0, 1))
        self.assertEqual(pairs[0][2], 1.0)     # recency 0 -> uniform weight

    def test_build_pairs_margin_excludes(self):
        from trainer.pb_irl import build_preference_pairs
        self.assertEqual(
            build_preference_pairs([1.0, 0.9], [0, 1], margin=0.5, recency=0.0), [])


class TestBuildPrefDataset(unittest.TestCase):
    def _make_dataset(self, n_days=8, N=50, D=6):
        # N>=40 so the expert's industry cap (int(target_k*0.3)) is >= 1
        rng = np.random.RandomState(0)
        ds = []
        for _ in range(n_days):
            ds.append({
                "features": torch.tensor(rng.randn(N, D), dtype=torch.float32),
                "labels": torch.tensor(rng.randn(N), dtype=torch.float32),
                "corr": torch.zeros(N, N),
                "industry_matrix": torch.zeros(N, N),
                "pos_matrix": torch.zeros(N, N),
                "neg_matrix": torch.zeros(N, N),
            })
        return ds

    def _args(self, margin):
        from types import SimpleNamespace
        return SimpleNamespace(market="t", pb_horizon=2, pb_margin=margin,
                               pb_recency=0.0, train_start="a", train_end="b",
                               ind_yn=False, pos_yn=False, neg_yn=False)

    def test_structure(self):
        import tempfile
        from trainer.pb_irl import build_preference_dataset
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            out = build_preference_dataset(self._args(0.0), ds, cache_dir=tmp)
        self.assertIn("traj_cache", out)
        self.assertIn("pairs", out)
        self.assertGreater(len(out["traj_cache"]), 0)
        self.assertGreater(len(out["pairs"]), 0)

    def test_no_pairs_raises(self):
        import tempfile
        from trainer.pb_irl import build_preference_dataset
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                build_preference_dataset(self._args(1e9), ds, cache_dir=tmp)


class TestCheckpointLoaders(unittest.TestCase):
    def test_strip_body_prefix(self):
        from trainer.pb_irl import strip_body_prefix
        sd = {"body.encoders.base.0.weight": 1, "body.weights": 2, "other": 3}
        out = strip_body_prefix(sd)
        self.assertEqual(set(out.keys()), {"encoders.base.0.weight", "weights"})

    def test_irl_state_dict_roundtrip(self):
        from trainer.irl_trainer import MultiRewardNetwork
        net = MultiRewardNetwork(input_dim=6, num_stocks=10, agg="sum")
        net2 = MultiRewardNetwork(input_dim=6, num_stocks=10, agg="mean")
        net2.load_state_dict(net.state_dict())  # must not raise

    def test_gail_body_loads_into_multireward(self):
        from trainer.irl_trainer import MultiRewardNetwork, GAILDiscriminator
        from trainer.pb_irl import strip_body_prefix
        disc = GAILDiscriminator(input_dim=6, num_stocks=10)
        net = MultiRewardNetwork(input_dim=6, num_stocks=10)
        net.load_state_dict(strip_body_prefix(disc.state_dict()))  # must not raise


class TestPBIRLTrainer(unittest.TestCase):
    def _setup(self, kl_on):
        from trainer.irl_trainer import MultiRewardNetwork
        from trainer.pb_irl import PBIRLTrainer
        torch.manual_seed(0)
        N, D = 10, 6
        net = MultiRewardNetwork(input_dim=D, num_stocks=N, agg="sum")
        s0 = np.random.RandomState(1).randn(N, D).astype(np.float32)
        s1 = np.random.RandomState(2).randn(N, D).astype(np.float32)
        a = np.ones(N, dtype=np.float32)
        pref = {"traj_cache": {0: (s0, a), 1: (s1, a)},
                "pairs": [(0, 1, 1.0)]}     # day 0 preferred over day 1
        return PBIRLTrainer(net, pref, kl_coef=1.0, kl_on=kl_on, n_pairs=1), net

    def test_step_updates_params(self):
        tr, net = self._setup(kl_on=True)
        before = [p.detach().clone() for p in net.parameters()]
        out = tr.train_step(device="cpu")
        after = list(net.parameters())
        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after)))
        self.assertIn("bt_loss", out)
        self.assertIn("pref_acc", out)

    def test_kl_grows_after_drift(self):
        tr, _ = self._setup(kl_on=True)
        tr.train_step(device="cpu")            # first step: params == prior -> kl 0
        out = tr.train_step(device="cpu")      # now drifted -> kl > 0
        self.assertGreater(out["kl"], 0.0)

    def test_kl_off_is_zero(self):
        tr, _ = self._setup(kl_on=False)
        tr.train_step(device="cpu")
        out = tr.train_step(device="cpu")
        self.assertEqual(out["kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
