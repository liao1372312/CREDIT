import torch
from credit.v3.outcome import Heads
from credit.v3_online_fast import SharedFeatureHeads


def test_shared_features_preserve_logits_likelihood_and_all_parameter_gradients():
    torch.manual_seed(43)
    old=Heads(4,feature_size=34,memory_size=32,hidden=32)
    new=SharedFeatureHeads(4,feature_size=34,memory_size=32,hidden=32)
    new.load_state_dict(old.state_dict())
    x=torch.randn(12,34)[torch.arange(96)%12]
    memory=torch.randn(9,32);pair=torch.randn(36,96)
    unary_mask=torch.randint(0,2,(96,9)).float();pair_mask=torch.randint(0,2,(96,36)).float()
    gate=torch.randint(0,8,(96,36));weight=torch.rand(96,2)
    a=old(x,memory,unary_mask,pair,pair_mask,gate)
    b=new(x,memory,unary_mask,pair,pair_mask,gate)
    assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    loss_a=(torch.nn.functional.softplus(a)*weight).mean()
    loss_b=(torch.nn.functional.softplus(b)*weight).mean()
    assert torch.allclose(loss_a,loss_b,atol=1e-6,rtol=1e-6)
    loss_a.backward();loss_b.backward()
    for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
        assert torch.allclose(p.grad,q.grad,atol=2e-6,rtol=2e-5),name
