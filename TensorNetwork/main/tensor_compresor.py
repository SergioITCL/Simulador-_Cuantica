import numpy as np
import torch
import tntorch
import tensorkrowch as tk


def compresor(tensor_train:list[tk.Node], max_rank:int):
    if max_rank is None:
        return tensor_train

    network = tensor_train[0].network

    # Creamos la forma TT en tntorch

    
    tensor_list = [ _.tensor for _ in tensor_train]

    TT = tntorch.Tensor(tensor_list)
   
    # Comprimimos el tensor train
    TT.round_tt(rmax=max_rank)

    # Recuperamos los tensores del TT
    TT_cores = [ tk.Node(tensor=core, shape=core.shape, name=f'intermedio_({_})', network=network, axes_names=tensor_train[_].axes_names)  for _, core in enumerate(TT.cores) ]
    for _ in tensor_train:
        network._remove_node(_)

    return TT_cores