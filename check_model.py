"""Small correctness checks; does not require the chest X-ray dataset."""
import torch
from experiment import VAE, objective

torch.set_num_threads(2)
torch.manual_seed(42)
net=VAE()
x=torch.rand(3,1,64,64)
logits,mu,lv=net(x)
assert logits.shape==x.shape and mu.shape==lv.shape==(3,32)
total,rec,kl=objective(logits,x,mu,lv)
assert torch.isfinite(total) and kl>=0
torch.testing.assert_close(total,rec+kl)
total.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
_,_,zero_kl=objective(logits,x,torch.zeros_like(mu),torch.zeros_like(lv))
assert zero_kl.item()==0
net.eval()
with torch.no_grad():
    torch.testing.assert_close(net(x,sample=False)[0],net(x,sample=False)[0])
print('PASS: tensor shapes, loss identity, finite gradients, zero standard-normal KL, deterministic mean reconstruction.')
