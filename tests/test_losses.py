import unittest
import torch
from src.losses import TrustAwareLoss, TrustAwareCPNLoss


class TestTrustLosses(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.dim = 8
        self.y_pred = torch.randn(self.batch_size, self.dim, requires_grad=True)
        self.y_true = torch.randn(self.batch_size, self.dim)
        self.variance = torch.rand(self.batch_size, self.dim) + 0.1
        self.u_control = torch.randn(self.batch_size, 4, requires_grad=True)

    def test_trust_aware_loss_forward_and_backward(self):
        loss_fn = TrustAwareLoss()
        loss = loss_fn(self.y_pred, self.y_true, self.variance)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))
        
        loss.backward()
        self.assertIsNotNone(self.y_pred.grad)

    def test_zero_variance_clamping_stability(self):
        loss_fn = TrustAwareLoss(eps=1e-6)
        zero_variance = torch.zeros(self.batch_size, self.dim)
        
        loss = loss_fn(self.y_pred, self.y_true, zero_variance)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_cpn_composite_loss(self):
        loss_fn = TrustAwareCPNLoss(lambda_u=0.05, lambda_trust=1.0)
        loss = loss_fn(self.y_pred, self.y_true, self.u_control, self.variance)
        
        loss.backward()
        self.assertIsNotNone(self.u_control.grad)


if __name__ == "__main__":
    unittest.main()
