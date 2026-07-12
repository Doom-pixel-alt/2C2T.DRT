import unittest

import numpy as np

import c2t as nn
from c2t.memory import GradientCheckpointer, estimate_activation_memory
from c2t.tensor import Tensor


def _copy_model(model):
    """Construct the small test architecture and copy all state into it."""
    result = nn.Sequential(
        nn.Dense(4, 6), nn.BatchNorm(6), nn.ReLU(), nn.Dropout(0.25), nn.Dense(6, 3)
    )
    result.load_state_dict(model.state_dict())
    return result


class MemoryTrainingTests(unittest.TestCase):
    def test_checkpoint_matches_normal_gradients_and_batchnorm_state(self):
        np.random.seed(11)
        template = nn.Sequential(
            nn.Dense(4, 6), nn.BatchNorm(6), nn.ReLU(), nn.Dropout(0.25), nn.Dense(6, 3)
        )
        normal = _copy_model(template)
        checkpointed = _copy_model(template)
        x_data = np.random.RandomState(7).randn(5, 4).astype(np.float32)
        targets = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        loss_fn = nn.CrossEntropyLoss()

        np.random.seed(42)
        normal_loss = loss_fn(normal(Tensor(x_data)), Tensor(targets))
        normal_loss.backward()

        np.random.seed(42)
        checkpointer = GradientCheckpointer(num_segments=3)
        output, state = checkpointer.checkpoint_forward(checkpointed, Tensor(x_data))
        checkpointed_loss = loss_fn(output, Tensor(targets))
        checkpointer.checkpoint_backward(checkpointed_loss, output, state)

        self.assertAlmostEqual(normal_loss.item(), checkpointed_loss.item(), places=7)
        for expected, actual in zip(normal.parameters(), checkpointed.parameters()):
            if expected.requires_grad:
                np.testing.assert_allclose(expected.grad, actual.grad, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(normal[1].running_mean.data, checkpointed[1].running_mean.data)
        np.testing.assert_allclose(normal[1].running_var.data, checkpointed[1].running_var.data)

    def test_accumulation_matches_a_true_large_batch(self):
        np.random.seed(123)
        template = nn.Sequential(nn.Dense(3, 4), nn.ReLU(), nn.Dense(4, 2))
        state = template.state_dict()
        accumulated = nn.Sequential(nn.Dense(3, 4), nn.ReLU(), nn.Dense(4, 2))
        large_batch = nn.Sequential(nn.Dense(3, 4), nn.ReLU(), nn.Dense(4, 2))
        accumulated.load_state_dict(state)
        large_batch.load_state_dict(state)

        x_data = np.random.randn(8, 3).astype(np.float32)
        targets = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int64)
        loss_fn = nn.CrossEntropyLoss()
        trainer_a = nn.Trainer(accumulated, loss_fn, nn.SGD(accumulated.parameters(), lr=0.1), verbose=False)
        trainer_b = nn.Trainer(large_batch, loss_fn, nn.SGD(large_batch.parameters(), lr=0.1), verbose=False)

        trainer_a.train_epoch(
            nn.DataLoader(nn.TensorDataset(x_data, targets), batch_size=2, shuffle=False),
            grad_accumulation=2,
        )
        trainer_b.train_epoch(
            nn.DataLoader(nn.TensorDataset(x_data, targets), batch_size=4, shuffle=False),
        )

        for expected, actual in zip(large_batch.parameters(), accumulated.parameters()):
            np.testing.assert_allclose(expected.data, actual.data, rtol=1e-6, atol=1e-7)

    def test_backward_releases_the_intermediate_graph(self):
        model = nn.Sequential(nn.Dense(4, 8), nn.ReLU(), nn.Dense(8, 3))
        loss = nn.CrossEntropyLoss()(model(Tensor(np.ones((2, 4), dtype=np.float32))), Tensor([0, 1]))
        loss.backward()
        self.assertIsNone(loss._ctx)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_activation_estimate_uses_the_live_model_graph(self):
        model = nn.Sequential(nn.Dense(4, 8), nn.ReLU(), nn.Dense(8, 3))
        estimate = estimate_activation_memory(model, (4,), batch_size=2)
        self.assertGreater(estimate["bytes"], 2 * 4 * 4)
        self.assertGreater(estimate["per_sample_megabytes"], 0)


if __name__ == "__main__":
    unittest.main()
