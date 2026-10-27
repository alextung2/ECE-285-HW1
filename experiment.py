"""Reproducible chest X-ray VAE: prepare, train, and evaluate.
Run python experiment.py --help for paths and options. No patient labels are used.
"""
import argparse, csv, hashlib, io, json, random, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

class VAE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.latent = latent
        layers=[]
        for a,b in zip([1,16,32,64],[16,32,64,128]):
            layers += [nn.Conv2d(a,b,4,2,1), nn.ReLU()]
        self.encoder=nn.Sequential(*layers,nn.Flatten())
        self.mu=nn.Linear(2048,latent)
        self.logvar=nn.Linear(2048,latent)
        self.fc=nn.Linear(latent,2048)
        layers=[]
        for a,b in zip([128,64,32,16],[64,32,16,1]):
            layers += [nn.ConvTranspose2d(a,b,4,2,1)]
            if b != 1: layers += [nn.ReLU()]
        self.decoder=nn.Sequential(*layers)
    def decode(self,z):
        return self.decoder(self.fc(z).reshape(-1,128,4,4))
    def forward(self,x,sample=True):
        h=self.encoder(x); mu=self.mu(h); lv=self.logvar(h)
        z=mu+torch.exp(0.5*lv)*torch.randn_like(mu) if sample else mu
        return self.decode(z),mu,lv

def objective(logits,x,mu,lv):
    rec=F.binary_cross_entropy_with_logits(logits,x,reduction='none').flatten(1).sum(1)
    kl=-0.5*(1+lv-mu.square()-lv.exp()).sum(1)
    return (rec+kl).mean(),rec.mean(),kl.mean()

def prepare(archive,cache):
    cache.mkdir(parents=True,exist_ok=True)
    manifest={}
    with zipfile.ZipFile(archive) as z:
        # Kaggle archive repeats its directory tree. Select one canonical copy.
        for split in ['train','val','test']:
            names=sorted(n for n in z.namelist() if n.startswith('chest_xray/'+split+'/') and n.lower().endswith(('.jpeg','.jpg','.png')) and '/._' not in n)
            if not names: raise ValueError('Expected chest_xray/'+split+' in archive')
            images=[]; rows=[]
            for name in names:
                raw=z.read(name)
                with Image.open(io.BytesIO(raw)) as im:
                    images.append(np.asarray(im.convert('L').resize((64,64),Image.Resampling.BILINEAR)))
                rows.append({'path':name,'class':Path(name).parent.name,'sha256':hashlib.sha256(raw).hexdigest()})
            np.save(cache/(split+'.npy'),np.stack(images))
            manifest[split]=rows
            print(split,len(images),flush=True)
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap={r['sha256'] for r in manifest[a]} & {r['sha256'] for r in manifest[b]}
        if overlap: raise ValueError(f'Exact file overlap between {a} and {b}: {len(overlap)}')

def load(cache,split):
    return torch.from_numpy(np.load(cache/(split+'.npy'))).unsqueeze(1).float()/255

def train(args):
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    x=load(args.cache,'train'); v=load(args.cache,'val').to(device)
    loader=DataLoader(TensorDataset(x),batch_size=64,shuffle=True,num_workers=0)
    net=VAE().to(device); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    config=vars(args).copy(); config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(latent_dim=32,batch_size=64,learning_rate=1e-3,beta=1,image_size=64,parameters=sum(p.numel() for p in net.parameters()),torch=torch.__version__)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    history=[]; best=float('inf'); start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        net.train(); sums=np.zeros(3)
        for (batch,) in loader:
            batch=batch.to(device); opt.zero_grad(set_to_none=True)
            logits,mu,lv=net(batch)
            loss,rec,kl=objective(logits,batch,mu,lv)
            loss.backward(); opt.step()
            sums+=np.array([loss.item(),rec.item(),kl.item()])*len(batch)
        net.eval()
        with torch.no_grad(),torch.random.fork_rng(devices=[] if device.type=='cpu' else [device]):
            torch.manual_seed(12345)
            val_draws=[]
            for _ in range(8):
                logits,mu,lv=net(v)
                val_draws.append([q.item() for q in objective(logits,v,mu,lv)])
            val=np.mean(val_draws,axis=0)
        row=dict(epoch=epoch,train_total=sums[0]/len(x),train_reconstruction=sums[1]/len(x),train_kl=sums[2]/len(x),val_total=val[0],val_reconstruction=val[1],val_kl=val[2],seconds=time.perf_counter()-start)
        history.append(row)
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row); w.writeheader(); w.writerows(history)
        if val[0]<best:
            best=val[0]; torch.save({'model':net.state_dict(),'epoch':epoch,'config':config},out/'best.pt')
        print(json.dumps(row),flush=True)
    print('Training complete',flush=True)

def evaluate(args):
    from torchvision.models import inception_v3,Inception_V3_Weights
    from ignite.metrics import FID, InceptionScore
    import torchvision, ignite
    torch.set_num_threads(args.threads); torch.manual_seed(2026)
    device=torch.device(args.device); out=args.out
    ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    net=VAE().to(device); net.load_state_dict(ckpt['model']); net.eval()
    real=load(args.cache,'test'); n=len(real)
    with torch.no_grad():
        fake=torch.cat([net.decode(torch.randn(min(64,n-i),32,device=device)).sigmoid().cpu() for i in range(0,n,64)])
        save_image(fake[:32],out/'generated.png',nrow=8,padding=2)
        save_image(torch.cat([real[:8],net(real[:8].to(device),sample=False)[0].sigmoid().cpu()]),out/'reconstructions.png',nrow=8,padding=2)
    np.save(out/'generated.npy',fake.numpy())
    inc=inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,transform_input=True).to(device).eval()
    captured=[]
    hook=inc.avgpool.register_forward_hook(lambda m,i,o:captured.append(o.flatten(1)))
    def features(images):
        fs=[]; ps=[]
        mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        with torch.inference_mode():
            for i in range(0,len(images),16):
                inp=F.interpolate(images[i:i+16].to(device).repeat(1,3,1,1),size=(299,299),mode='bilinear',align_corners=False)
                captured.clear(); logits=inc((inp-mean)/std)
                fs.append(captured[0].cpu()); ps.append(logits.softmax(1).cpu())
                if i%160==0: print('Inception',i,'/',len(images),flush=True)
        return torch.cat(fs),torch.cat(ps)
    rf,rp=features(real); ff,fp=features(fake); hook.remove()
    np.savez_compressed(out/'inception_features.npz',real_features=rf.numpy(),fake_features=ff.numpy(),real_probabilities=rp.numpy(),fake_probabilities=fp.numpy())
    fid=FID(num_features=2048,feature_extractor=nn.Identity())
    fid.update((ff,rf))
    score=InceptionScore(num_features=1000,feature_extractor=nn.Identity()); score.update(fp)
    real_score=InceptionScore(num_features=1000,feature_extractor=nn.Identity()); real_score.update(rp)
    splits=[]
    for part in fp.tensor_split(10):
        m=InceptionScore(num_features=1000,feature_extractor=nn.Identity()); m.update(part); splits.append(m.compute())
    with torch.no_grad():
        sums=np.zeros(3)
        for i in range(0,n,64):
            b=real[i:i+64].to(device); logits,mu,lv=net(b)
            sums+=np.array([q.item() for q in objective(logits,b,mu,lv)])*len(b)
    fid_value=fid.compute(); is_value=score.compute()
    # Independent, low-rank formula checks numerical stability of singular covariances.
    a=rf.double()-rf.double().mean(0); b=ff.double()-ff.double().mean(0)
    cross_trace=torch.linalg.svdvals(a@b.T).sum()/(n-1)
    fid_check=((rf.double().mean(0)-ff.double().mean(0)).square().sum()+(a.square().sum()+b.square().sum())/(n-1)-2*cross_trace).item()
    probs=fp.double(); is_check=(probs*(probs.clamp_min(1e-16).log()-probs.mean(0).clamp_min(1e-16).log())).sum(1).mean().exp().item()
    assert abs(fid_value-fid_check)<0.02, (fid_value,fid_check)
    assert abs(is_value-is_check)<1e-5, (is_value,is_check)
    results={'checkpoint_epoch':ckpt['epoch'],'n_real':n,'n_generated':n,'fid_imagenet_pool2048':fid_value,'fid_independent_check':fid_check,'inception_score_global':is_value,'is_independent_check':is_check,'inception_score_split_mean':float(np.mean(splits)),'inception_score_split_std':float(np.std(splits)),'real_inception_score_global':real_score.compute(),'test_total':sums[0]/n,'test_reconstruction':sums[1]/n,'test_kl':sums[2]/n,'torchvision':torchvision.__version__,'ignite':ignite.__version__,'generation_seed':2026,'metric_protocol':'torchvision ImageNet Inception-v3; grayscale repeated to RGB; bilinear resize 299; ImageNet normalization; transform_input=True; pool3 FID; global IS and 10 contiguous splits; 624 samples, no class balancing'}
    (out/'metrics.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','train','evaluate']); p.add_argument('--archive',type=Path,default=Path('chest-xray.zip')); p.add_argument('--cache',type=Path,default=Path('cache')); p.add_argument('--out',type=Path,default=Path('results')); p.add_argument('--epochs',type=int,default=30); p.add_argument('--seed',type=int,default=42); p.add_argument('--threads',type=int,default=4); p.add_argument('--device',default='cpu'); a=p.parse_args()
    if a.stage=='prepare': prepare(a.archive,a.cache)
    elif a.stage=='train': train(a)
    else: evaluate(a)
