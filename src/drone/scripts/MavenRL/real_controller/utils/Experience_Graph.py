
import numpy as np
import copy
from rllib.template.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data as Graph_Data
from torch_geometric.data.batch import Batch


################################


def generate_fully_connected_graph_with_edge_attr_with_formation_aware(node_feature, candidate_edge_feature):
    agent_nums = node_feature.shape[0]
    # state = state.reshape(-1, params.NUM_AGENT - 1, 4)
    # ego = torch.zeros(state.shape[0], 1, 4).cuda()
    # state = torch.cat([ego, state], dim=1) 
    # startnode_index = [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(1,agent_nums)].append(0)
    startnode_index = []
    endnode_index = []
    candidate_edge_feature = torch.from_numpy(candidate_edge_feature).float()
    edge_feature = torch.zeros((0,candidate_edge_feature.shape[1]+1))
    for i in range(agent_nums):
        for j in range(agent_nums):
            startnode_index.append(i)
            endnode_index.append(j)
            if abs(i-j)<=1 or (i==0 and j==agent_nums-1) or (j==0 and i==agent_nums-1):
                edge_feature = torch.vstack((edge_feature, torch.cat((candidate_edge_feature[i, : ] - candidate_edge_feature[j, : ], torch.tensor([1])),dim=0)  ))
            else:
                edge_feature = torch.vstack((edge_feature, torch.cat((candidate_edge_feature[i, : ] - candidate_edge_feature[j, : ], torch.tensor([0])),dim=0)  ))

    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=edge_feature)


def generate_egocentric_graph_with_edge_attr_with_formation_aware(node_feature, candidate_edge_feature, ego_id):
    agent_nums = node_feature.shape[0]
    # state = state.reshape(-1, params.NUM_AGENT - 1, 4)
    # ego = torch.zeros(state.shape[0], 1, 4).cuda()
    # state = torch.cat([ego, state], dim=1) 
    # startnode_index = [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(1,agent_nums)].append(0)
    # startnode_index = [ego_id for _ in range(agent_nums)] + [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(agent_nums)] + [ego_id for _ in range(agent_nums)]
    startnode_index = [i for i in range(agent_nums)]
    endnode_index = [ego_id for _ in range(agent_nums)]
    candidate_edge_feature = torch.from_numpy(candidate_edge_feature).float()
    edge_feature = torch.zeros((0,candidate_edge_feature.shape[1]+1))
    for i in range(agent_nums):
        # edge_feature = torch.vstack((edge_feature, candidate_edge_feature[i, : ] - candidate_edge_feature[ego_id, : ]))

        if abs(i-ego_id)<=1 or (i==0 and ego_id==agent_nums-1) or (ego_id==0 and i==agent_nums-1):
            edge_feature = torch.vstack((edge_feature, torch.cat((candidate_edge_feature[i, : ] - candidate_edge_feature[ego_id, : ], torch.tensor([1])),dim=0)  ))
        else:
            edge_feature = torch.vstack((edge_feature, torch.cat((candidate_edge_feature[i, : ] - candidate_edge_feature[ego_id, : ], torch.tensor([0])),dim=0)  ))

    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=edge_feature)



#######################################
def generate_egocentric_graph(node_feature, ego_id):
    agent_nums = node_feature.shape[0]
    # state = state.reshape(-1, params.NUM_AGENT - 1, 4)
    # ego = torch.zeros(state.shape[0], 1, 4).cuda()
    # state = torch.cat([ego, state], dim=1) 
    # startnode_index = [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(1,agent_nums)].append(0)
    # startnode_index = [ego_id for _ in range(agent_nums)] + [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(agent_nums)] + [ego_id for _ in range(agent_nums)]
    startnode_index = [i for i in range(agent_nums)]
    endnode_index = [ego_id for _ in range(agent_nums)]

    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=None)


def generate_egocentric_graph_with_edge_attr(node_feature, candidate_edge_feature, ego_id):
    agent_nums = node_feature.shape[0]
    # state = state.reshape(-1, params.NUM_AGENT - 1, 4)
    # ego = torch.zeros(state.shape[0], 1, 4).cuda()
    # state = torch.cat([ego, state], dim=1) 
    # startnode_index = [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(1,agent_nums)].append(0)
    # startnode_index = [ego_id for _ in range(agent_nums)] + [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(agent_nums)] + [ego_id for _ in range(agent_nums)]
    startnode_index = [i for i in range(agent_nums)]
    endnode_index = [ego_id for _ in range(agent_nums)]
    candidate_edge_feature = torch.from_numpy(candidate_edge_feature).float()
    edge_feature = torch.zeros((0,candidate_edge_feature.shape[1]))
    for i in range(agent_nums):
        edge_feature = torch.vstack((edge_feature, candidate_edge_feature[i, : ] - candidate_edge_feature[ego_id, : ]))

    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=edge_feature)

def generate_fully_connected_graph(node_feature):
    agent_nums = node_feature.shape[0]
    # state = state.reshape(-1, params.NUM_AGENT - 1, 4)
    # ego = torch.zeros(state.shape[0], 1, 4).cuda()
    # state = torch.cat([ego, state], dim=1) 
    # startnode_index = [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(1,agent_nums)].append(0)
    startnode_index = []
    endnode_index = []
    for i in range(agent_nums):
        for j in range(agent_nums):
            startnode_index.append(i)
            endnode_index.append(j)
    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=None)

def generate_fully_connected_graph_with_edge_attr(node_feature, candidate_edge_feature):
    agent_nums = node_feature.shape[0]
    # state = state.reshape(-1, params.NUM_AGENT - 1, 4)
    # ego = torch.zeros(state.shape[0], 1, 4).cuda()
    # state = torch.cat([ego, state], dim=1) 
    # startnode_index = [i for i in range(agent_nums)]
    # endnode_index = [i for i in range(1,agent_nums)].append(0)
    startnode_index = []
    endnode_index = []
    candidate_edge_feature = torch.from_numpy(candidate_edge_feature).float()
    edge_feature = torch.zeros((0,candidate_edge_feature.shape[1]))
    for i in range(agent_nums):
        for j in range(agent_nums):
            startnode_index.append(i)
            endnode_index.append(j)
            edge_feature = torch.vstack((edge_feature, candidate_edge_feature[i, : ] - candidate_edge_feature[j, : ]))
    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=edge_feature)

def generate_ring_graph(node_feature):
    agent_nums = node_feature.shape[0]
    # import pdb;pdb.set_trace()
    startnode_index = [i for i in range(agent_nums)] + [i for i in range(1,agent_nums)]+[0] + [i for i in range(agent_nums)]
    endnode_index = [i for i in range(1,agent_nums)]+[0] + [i for i in range(agent_nums)] + [i for i in range(agent_nums)]    
    edge = torch.tensor([
            startnode_index,
            endnode_index
        ])
    
    return Graph_Data(x=torch.tensor(node_feature).float(), edge_index=edge, edge_attr=None)


def data_graph_to_cuda(data_graph):
    data_graph.x = data_graph.x.cuda() 
    data_graph.edge_index = data_graph.edge_index.cuda()
    if  data_graph.edge_attr is not None:
        data_graph.edge_attr = data_graph.edge_attr.cuda()
    return data_graph

class Experience_Graph(Data):
    
    # =============================================================================
    # -- dict ---------------------------------------------------------------------
    # =============================================================================
    
    ###override for GNN data
    def to(self, *args, **kwargs):
        """
            for torch.Tensor
        """
        new_dict = dict()
        for (key, value) in self.__dict__.items():

            if isinstance(value, Data):
                new_dict[key] = value.to(*args, **kwargs)
            elif isinstance(value, torch.Tensor):
                new_dict[key] = value.to(*args, **kwargs)
            elif isinstance(value, list):
                new_dict[key] = [v.to(*args, **kwargs) for v in value]
            elif key == "state_graph" or key == "next_state_graph":
                temp_list = []
                for data_graph in self.__dict__[key]:
                    data_graph = data_graph_to_cuda(data_graph)
                    temp_list.append(data_graph)
                new_dict[key] = Batch.from_data_list(temp_list)
        
            elif key == "cen_state_graph" or key == "next_cen_state_graph" or key == "action_graph" or  key == "dec_state_graph" or key == "next_dec_state_graph":
                temp_list = []
                for data_graph in self.__dict__[key]:
                    data_graph = data_graph_to_cuda(data_graph)
                    temp_list.append(data_graph)
                new_dict[key] = Batch.from_data_list(temp_list)
            else:
                raise NotImplementedError
                # new_dict[key] = 'NotImplementedError'
        return type(self)(**new_dict)

    ###override for GNN data
    def cat(self, *args, **kwargs):
        """
            for torch.Tensor
        """
        # import pdb;pdb.set_trace()
        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.cat(*args, **kwargs)
            # elif isinstance(value, torch.Tensor):
            elif key == "state_graph" or key == "next_state_graph":
                new_dict[key] = value
            elif key == "cen_state_graph" or key == "next_cen_state_graph" or key == "action_graph":
                new_dict[key] = value
            elif key == "dec_state_graph" or key == "next_dec_state_graph":
                temp_list = []
                for graph_list in value:
                    temp_list += graph_list 
                new_dict[key] = tuple(temp_list)
            elif all([isinstance(v, torch.Tensor) for v in value]):
                try:
                    new_dict[key] = torch.cat(value, *args, **kwargs)#torch.cat(tuple(), *args, **kwargs)
                except:
                    import pdb;pdb.set_trace()
            else:
                new_dict[key] = torch.as_tensor(value)
        return type(self)(**new_dict)


    def stack(self, *args, **kwargs):
        """
            for torch.Tensor
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.stack(*args, **kwargs)
            elif all([isinstance(v, torch.Tensor) for v in value]):
                new_dict[key] = torch.stack(value, *args, **kwargs)
            else:
                new_dict[key] = torch.as_tensor(value)
        return type(self)(**new_dict)




    
    def to_tensor(self):
        """
            for np.ndarray
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, np.ndarray):
                new_dict[key] = torch.from_numpy(value)
            elif isinstance(value, Data):
                new_dict[key] = value.to_tensor()
            else:
                # raise NotImplementedError
                new_dict[key] = torch.tensor(value)
        return type(self)(**new_dict)



