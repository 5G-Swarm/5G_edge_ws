
import numpy as np
import copy
import torch
from rllib.template.data import Data

class Data_Graph(Data):
    
    # =============================================================================
    # -- dict ---------------------------------------------------------------------
    # =============================================================================

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
            elif key == "graph":
                new_dict[key] = value.to(*args, **kwargs)
            else:
                raise NotImplementedError
                # new_dict[key] = 'NotImplementedError'
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
            elif key == "graph":
                new_dict[key] = value
            elif all([isinstance(v, torch.Tensor) for v in value]):
                new_dict[key] = torch.cat(value, *args, **kwargs)#torch.cat(tuple(), *args, **kwargs)
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



