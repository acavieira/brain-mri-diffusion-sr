import torch

from mri_diffusion.diffusion import Diffusion


class Oracle(torch.nn.Module):
    """Returns the exact v of x_t for a known clean image x0."""

    def __init__(self, diffusion, x0):
        super().__init__()
        self.diffusion, self.x0 = diffusion, x0

    def forward(self, x_t, condition, t):
        a = self.diffusion.alpha_bar[t].view(-1, 1, 1, 1)
        # from x_t = sqrt(a) x0 + sqrt(1 - a) eps and v = sqrt(a) eps - sqrt(1 - a) x0
        return (a.sqrt() * x_t - self.x0) / (1 - a).sqrt()


def test_alpha_bar_goes_from_almost_one_to_almost_zero():
    d = Diffusion()
    assert torch.all(d.alpha_bar[1:] < d.alpha_bar[:-1])
    assert d.alpha_bar[0] > 0.999 and d.alpha_bar[-1] < 1e-3


def test_add_noise_keeps_x0_at_the_first_step():
    d = Diffusion()
    x0 = torch.rand(2, 1, 8, 8)
    x_t = d.add_noise(x0, torch.randn_like(x0), torch.zeros(2, dtype=torch.long))
    # at t=0 the noise has a standard deviation of sqrt(1e-4) = 0.01
    assert torch.allclose(x_t, x0, atol=0.06)


def test_x0_comes_back_from_x_t_and_v():
    d = Diffusion()
    x0, noise = torch.rand(4, 1, 8, 8) * 2 - 1, torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 10, 500, 999])
    x_t, v = d.add_noise(x0, noise, t), d.velocity(x0, noise, t)
    a = d.alpha_bar[t].view(-1, 1, 1, 1)
    assert torch.allclose(a.sqrt() * x_t - (1 - a).sqrt() * v, x0, atol=1e-5)


def test_sampler_recovers_x0_with_a_perfect_model():
    d = Diffusion()
    x0 = torch.rand(2, 1, 8, 8) * 2 - 1
    for steps, eta in [(20, 0.0), (1000, 1.0), (10, 0.5)]:
        out = d.sample(Oracle(d, x0), torch.zeros_like(x0), steps=steps, eta=eta,
                       generator=torch.Generator().manual_seed(0))
        assert torch.allclose(out, x0, atol=1e-4), (steps, eta)


def test_loss_of_a_zero_model_is_the_mean_square_of_v():
    d = Diffusion()
    model = lambda x_t, condition, t: torch.zeros_like(x_t)  # noqa: E731
    x0, noise = torch.rand(4, 1, 8, 8), torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 10, 500, 999])
    loss = d.loss(model, x0, x0, t=t, noise=noise)
    assert loss.ndim == 0
    assert torch.isclose(loss, d.velocity(x0, noise, t).pow(2).mean())
