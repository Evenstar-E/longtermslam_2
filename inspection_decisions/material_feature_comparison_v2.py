"""Neural-only feature extraction; identical measured samples as V1 controls."""
import numpy as np
import torch


@torch.inference_mode()
def learned_epochs(pieces,models,geometry_epochs):
    result={name:[] for name in models}
    for e,views in enumerate(pieces):
        buckets={name:[dict(xyz=[],desc=[]) for _ in range(24)] for name in models}
        for view in views:
            x=torch.from_numpy(view['patch'])
            for name,model in models.items():
                desc=np.concatenate([model(a).numpy() for a in x.split(512)]) if len(x) else np.zeros((0,64),np.float32)
                for slot in np.unique(view['slots']):
                    take=view['slots']==slot;b=buckets[name][slot]
                    b['xyz'].append(view['xyz'][take]);b['desc'].append(desc[take])
        for name in models:
            ep=[]
            for slot in range(24):
                b=buckets[name][slot];g=geometry_epochs[e][slot]
                ep.append(dict(points=g['points'],tree=g['tree'],xyz=np.concatenate(b['xyz']) if b['xyz'] else np.zeros((0,3)),
                    desc=np.concatenate(b['desc']) if b['desc'] else np.zeros((0,64),np.float32)))
            result[name].append(ep)
    return result
