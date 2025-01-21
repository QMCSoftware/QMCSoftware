import numpy as np 
import os 
    
def train_val_split(R, to_split, val_frac=1/8, shuffle=True, rng_shuffle_seed=None):
    if isinstance(to_split[0],np.ndarray):
        npt = np 
    else:
        import torch 
        assert isinstance(to_split[0],torch.Tensor), "to_split list must contain np.ndarray or torch.Tensor items"
        npt = torch 
    if shuffle:
        rng = np.random.Generator(np.random.PCG64(rng_shuffle_seed))
        tv_idx = rng.permutation(R)
        if npt!=np: tv_idx = torch.from_numpy(tv_idx)
    else:
        tv_idx = npt.arange(R)
    n_train = R-int(val_frac*R)
    n_val = R-n_train
    tidx = tv_idx[:n_train]
    vidx = tv_idx[n_train:]
    splits = (vval for val in to_split for vval in (val[tidx],val[vidx]))
    return splits 

def parse_metrics(path):
    import pandas as pd
    metrics = pd.read_csv(path)
    tags = [col[6:] for col in metrics.columns if "train_" in col]
    metrics_train = metrics.iloc[~np.isnan(metrics["train_"+tags[0]].values)]
    metrics_val = metrics.iloc[~np.isnan(metrics["val_"+tags[0]].values)]
    parsed_metrics = {}
    for tag in tags:
        parsed_metrics["train_"+tag] = metrics_train["train_"+tag].values
        parsed_metrics["val_"+tag] = metrics_val["val_"+tag].values
    parsed_metrics = pd.DataFrame(parsed_metrics)
    parsed_metrics["epoch"] = np.arange(metrics["epoch"][0],metrics["epoch"][0]+len(parsed_metrics))
    newpath = path[:-4]+"_parsed.csv"
    if os.path.isfile(newpath):
        parsed_metrics_old = pd.read_csv(newpath)
        if parsed_metrics_old["epoch"][len(parsed_metrics_old)-1]==(parsed_metrics["epoch"][0]-1): # append
            parsed_metrics = pd.concat([parsed_metrics_old,parsed_metrics])
    parsed_metrics.reset_index(drop=True,inplace=True)
    parsed_metrics.to_csv(newpath,index=False)
    return parsed_metrics.drop('epoch',axis=1)

def plot_metrics(metrics, tags=None, logscale=None, s0=0, linewidth=3, color_train=None, color_val=None):
    from matplotlib import pyplot
    if tags is None: tags = [col[6:] for col in metrics.columns if "train_" in col]
    fig,ax = pyplot.subplots(nrows=1,ncols=len(tags),figsize=(7*len(tags),5))
    ax = np.atleast_1d(ax)
    assert ax.ndim==1
    epochs = metrics.index[s0:]
    for i,tag in enumerate(tags):
        tvals = metrics["train_"+tag][s0:]
        vvals = metrics["val_"+tag][s0:]
        ax[i].set_ylabel(tag)
        ax[i].plot(epochs,tvals,label="train",linewidth=linewidth,color=color_train)
        ax[i].plot(epochs,vvals,label="val",linewidth=linewidth,color=color_val)
        ax[i].set_xlabel("epoch")
        ax[i].legend()
        if (logscale is True) or (isinstance(logscale,list) and logscale[i]) or (logscale is None and (tvals>0).all() and (vvals>0).all()):
            ax[i].set_yscale("log",base=10)
    return fig,ax