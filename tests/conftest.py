"""Offline tests: keep model checks on CPU and use small synthetic inputs."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
import torch
torch.set_num_threads(1)
