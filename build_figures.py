"""Regenerate report assets from measured experiment outputs."""
import csv,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1]
results=base/'results'; report=base/'report'
history=list(csv.DictReader((results/'history.csv').open()))
history=[{k:float(v) for k,v in r.items()} for r in history]
metrics=json.loads((results/'metrics.json').read_text())
config=json.loads((results/'config.json').read_text())
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(7.2,1.65),layout='constrained')
for ax,term,title in zip(axs,['total','reconstruction','kl'],['Total loss','Reconstruction BCE','KL divergence']):
    for split,color in [('train','#245ca6'),('val','#ba452e')]:
        ax.plot([h['epoch'] for h in history],[h[split+'_'+term] for h in history],label=split,color=color,lw=1.3)
    ax.axvline(metrics['checkpoint_epoch'],ls='--',color='0.5',lw=.8)
    ax.set(xlabel='Epoch',title=title); ax.grid(alpha=.15)
axs[0].set_ylabel('Nats / image'); axs[0].legend(frameon=False,fontsize=7)
fig.savefig(report/'loss_curves.png',dpi=240); plt.close(fig)
shutil.copy2(results/'generated.png',report/'generated.png')
values={'FID':f"{metrics['fid_imagenet_pool2048']:.2f}",'IS':f"{metrics['inception_score_global']:.3f}",'SplitIS':f"{metrics['inception_score_split_mean']:.3f} \\pm {metrics['inception_score_split_std']:.3f}",'RealIS':f"{metrics['real_inception_score_global']:.3f}",'TestRec':f"{metrics['test_reconstruction']:.2f}",'TestKL':f"{metrics['test_kl']:.2f}",'TestLoss':f"{metrics['test_total']:.2f}",'Params':f"{config['parameters']:,}",'BestEpoch':str(metrics['checkpoint_epoch']),'Minutes':f"{history[-1]['seconds']/60:.1f}"}
values['SplitIS']='$'+values['SplitIS']+'$'
(report/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in values.items()))
print(json.dumps(values,indent=2))
