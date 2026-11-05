"""Track unknown conditional-input mass on reachable finite workflow paths.

Intervals condition on fitted known kernels. They are not confidence bounds
for their estimation error or a certificate for arbitrary semantic services.
"""

import numpy as np

from .budget_mechanisms_v2 import BatchForecast
from .stateful import ACTIONS, NEIGHBORS, ROLES, STATUSES, decision_key


class PartialSupportForecast:
    def __init__(self, model):
        self.model = model
        self.batch = BatchForecast(model)
        self.policy_known = {}
        for role in ROLES:
            table = np.zeros((1 << len(NEIGHBORS[role]),8,4),dtype=bool)
            for mask in range(len(table)):
                exposure = [m for i,m in enumerate(NEIGHBORS[role]) if mask & (1 << i)]
                for state in range(8):
                    for index,previous in enumerate(STATUSES):
                        key = decision_key(role,state,previous,exposure)
                        table[mask,state,index] = model.policy.get(key,np.zeros(4)).sum() >= model.min_support
            self.policy_known[role] = table
        self.api_known = {}
        for epoch,family,role,state,action in model.api:
            key = epoch,family,role
            if key not in self.api_known:
                self.api_known[key] = np.array([[sum(model.api.get((*key,s,a),{}).values()) >= model.min_support
                    for a in ACTIONS] for s in range(8)],dtype=bool)

    def predict(self,task,exposures):
        count = len(exposures); distribution = np.zeros((count,33))
        distribution[:,32 if task.initial_state & 8 else task.initial_state*4] = 1
        for role in task.roles:
            terminal = distribution[:,32].copy()
            key = task.epoch,task.family,role
            if role not in ROLES or key not in self.api_known:
                distribution[:,:32] = 0
                continue
            masks = np.array([sum(1 << i for i,m in enumerate(NEIGHBORS[role]) if m in e) for e in exposures])
            state_mass = distribution[:,:32].reshape(-1,8,4) * self.policy_known[role][masks]
            action_mass = np.einsum('bsh,bsha->bsa',state_mass,self.batch.actions[role][masks])
            action_mass *= self.api_known[key]
            distribution = np.einsum('bsa,sad->bd',action_mass,self.batch.responses[key])
            distribution[:,32] += terminal
        known_success = distribution[:,32]
        unknown = np.clip(1-distribution.sum(axis=1),0,1)
        return {'known_success':known_success,'unknown_mass':unknown,
            'conditional_upper':np.minimum(1,known_success+unknown),
            'complete_path_coverage':unknown <= 1e-10,
            'scope':'bounds conditional on fitted known kernels, not estimation confidence certificates'}
