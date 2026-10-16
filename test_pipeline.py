import unittest
import numpy as np
import torch
from train import SmallCNN, split_data

class PipelineTests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_repeatable(self):
        x, y, train, val, test = split_data(42)
        all_indices = np.concatenate([train, val, test])
        self.assertEqual(len(np.unique(all_indices)), len(y))
        self.assertEqual(len(all_indices), len(y))
        self.assertTrue(np.array_equal(train, split_data(42)[2]))
        self.assertEqual(tuple(x.shape[1:]), (1, 8, 8))
        self.assertGreaterEqual(x.min().item(), 0)
        self.assertLessEqual(x.max().item(), 1)

    def test_logits_and_gradients(self):
        model = SmallCNN()
        logits = model(torch.rand(4, 1, 8, 8))
        self.assertEqual(tuple(logits.shape), (4, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__ == '__main__':
    unittest.main()
