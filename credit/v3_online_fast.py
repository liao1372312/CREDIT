"""Equivalent shared-feature neural computation for chronological refitting.

Feature vectors and content encodings are fixed tensors, not learned inputs.
The neural functions are evaluated once per identical feature vector, then
gathered before applying each exposure mask and role-specific interaction gate.
No likelihood, parameter, target, or optimizer setting changes.
"""
import torch

from .v3.outcome import Heads
from .v3_online import OnlineModel as OriginalOnlineModel,fit_ensemble as original_fit
from .v3_online import load_ensemble as original_load
from .v3_revalidation import candidate_memories
from .v3_benchmark import CachedModel
from .v3.audit import SavedCalibration
from .v3.experiment import write


class SharedFeatureHeads(Heads):
    def forward(self,x,memories,unary_mask,pair_features,pair_mask,gate_index,*,baseline_only=False):
        if x.requires_grad:raise ValueError('shared-feature compression expects fixed inputs')
        unique,inverse=torch.unique(x,dim=0,return_inverse=True)
        eta=self.baseline(unique)[inverse]
        if baseline_only:return eta
        n,m=len(unique),len(memories)
        features=torch.cat((unique[:,None,:].expand(n,m,-1),memories[None].expand(n,m,-1)),dim=-1)
        eta=eta+(self.unary(features)[inverse]*unary_mask[:,:,None]).sum(dim=1)
        if len(pair_features):
            p=len(pair_features)
            features=torch.cat((unique[:,None,:].expand(n,p,-1),pair_features[None].expand(n,p,-1)),dim=-1)
            eta=eta+(self.pair(features)[inverse]*self.gates[gate_index]*pair_mask[:,:,None]).sum(dim=1)
        return eta


class FastOnlineModel(OriginalOnlineModel):
    def __init__(self,pool,contexts,*,seed=0):
        super().__init__(pool,contexts,seed=seed)
        torch.manual_seed(seed)
        self.heads=SharedFeatureHeads(len(self.roles),feature_size=34,memory_size=32,hidden=32)


def fit_ensemble(train,validation,calibration,seed,directory,*,logged=(),members=2):
    from collections import defaultdict
    import numpy as np
    directory.mkdir();groups=defaultdict(list)
    for e in train:groups[e.context.uid].append(e)
    ids=sorted(groups);rng=np.random.default_rng(seed);models=[]
    for i in range(members):
        sample=[e for uid in rng.choice(ids,len(ids),replace=True) for e in groups[uid]]
        model=FastOnlineModel(candidate_memories(),[e.context for e in train],seed=seed+i)
        model.fit_online(sample,validation,calibration,logged=logged)
        model.fit_report['neural_computation']='shared fixed feature vectors; same likelihood and gradients within floating precision'
        model.save(directory/f'outcome_model_{i}.pt');models.append(CachedModel(model))
        print(seed,directory.name,i,'epochs',model.fit_report['iterations'],flush=True)
    write(directory/'training_reports.json',[m.fit_report for m in models]);return tuple(models)


def load_ensemble(directory,contexts):
    models=[]
    for path in sorted(directory.glob('outcome_model_*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        model=FastOnlineModel(candidate_memories(),contexts)
        if model.roles!=data['role_indices'] or [m.uid for m in model.memories]!=data['memory_ids']:raise ValueError('fast model registry differs')
        model.heads.load_state_dict(data['state_dict']);model.calibrator=SavedCalibration(data['calibration_x'],data['calibration_y'])
        model.fit_report=data['fit_report'];models.append(CachedModel(model))
    if len(models)!=2:raise ValueError('two checkpoints required')
    return tuple(models)
