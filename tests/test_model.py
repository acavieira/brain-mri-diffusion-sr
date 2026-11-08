import torch

from mri_diffusion.model import Attention, UNet


def small_unet(dropout=0.0):
    return UNet(base=32, mults=(1, 2, 2), res_blocks=1, attention_res=(16,),
                dropout=dropout, image_size=64)


def test_output_has_the_shape_of_the_input():
    model = small_unet()
    x = torch.randn(3, 1, 64, 64)
    out = model(x, torch.randn_like(x), torch.tensor([0, 500, 999]))
    assert out.shape == x.shape


def test_attention_only_at_the_requested_resolution():
    model = small_unet()
    # 16x16 is the last level: one encoder block, one decoder block and the middle
    assert sum(isinstance(m, Attention) for m in model.modules()) == 3


def test_dropout_is_off_in_eval_mode():
    model = small_unet(dropout=0.5).eval()
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([10, 20])
    with torch.no_grad():
        assert torch.equal(model(x, x, t), model(x, x, t))


def test_condition_changes_the_output():
    model = small_unet().eval()
    x = torch.randn(1, 1, 64, 64)
    t = torch.tensor([100])
    with torch.no_grad():
        assert not torch.allclose(model(x, torch.zeros_like(x), t), model(x, torch.ones_like(x), t))
