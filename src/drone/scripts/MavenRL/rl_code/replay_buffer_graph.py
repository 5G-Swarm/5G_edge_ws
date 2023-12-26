
import numpy as np
from typing import List

import torch
import dgl
from real_controller.utils.Experience_Graph import Experience_Graph as Experience
from rllib.buffer.replay_buffer import ReplayBuffer

def stack_data(datas: List[Experience]):
    data_keys = datas[0].keys()
    result = {}

    # temp = list(zip(data_keys, zip(*datas)))
    # import pdb;pdb.set_trace()
    for key, i in zip(data_keys, zip(*datas)):
        if isinstance(i[0], Experience):
            result[key] = stack_data(i)
        else:
            result[key] = i
    result = Experience(**result)
    return result

class ReplayBuffer_Graph(ReplayBuffer):
    def __init__(self, config, capacity, batch_size, device):
        super(ReplayBuffer_Graph, self).__init__(config, capacity, batch_size, device)
        self.memory = np.empty(self.capacity, dtype=Experience)

    def sample(self):
        # import pdb;pdb.set_trace()
        batch: List[Experience] = self.get_batch(self.batch_size)
        experiences: Experience = self._batch_stack(batch)
        num_agent_record = []
        for i in range(self.batch_size):
            num_agent_record.append(batch[i].num_agent)
        return experiences.to(self.device),num_agent_record
    
    def _batch_stack(self, batch):
        """
            To be override.
        """
        result = stack_data(batch)
        # result.update(graph=dgl.batch(list(result.graph)))
        result.update(reward=[*torch.tensor(result.reward, dtype=torch.float32).unsqueeze(1)])
        result.update(done=[*torch.tensor(result.done, dtype=torch.float32).unsqueeze(1)])
        result = result.cat(dim=0)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result

