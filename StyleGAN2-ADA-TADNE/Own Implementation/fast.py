from training.networks import Generator
import pickle
import torch
import numpy as np
#import matplotlib.pyplot as plt
from training.networks import Generator, Discriminator  # ensure the new repo is part of your python path
import PIL.Image
import logging
# your trained network file 
path = r"C:\Users\Aravind\Desktop\stylegan2-ada-pytorch-main\stylegan2-ada-pytorch-main\network-tadne.pt.pkl"


mapping_kwargs  = {"num_layers":4,"layer_features":1024}   # Arguments for MappingNetwork.
synthesis_kwargs    = {"channel_base":(32<<10)*2,"channel_max":1024}


with open(path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cpu()  # torch.nn.Module
    f.seek(0)
    #D = pickle.load(f)['D'].cpu()  # torch.nn.Module

# create a new network using the new defintion
G2 = Generator(
        z_dim = 1024,  # Input latent (Z) dimensionality.
        c_dim = 0,  # Conditioning label (C) dimensionality.
        w_dim = 1024,  # Intermediate latent (W) dimensionality.
        img_resolution = 512,  # Output resolution.
        img_channels = 3,
        mapping_kwargs=mapping_kwargs,
        synthesis_kwargs=synthesis_kwargs,

).cpu()


# update the weights to match your trained model
g_sd = G.state_dict()
g2_sd = G2.state_dict()

for k, _ in g2_sd.items():
    g2_sd[k] = g_sd[k]


G2.load_state_dict(g2_sd)

# import functools
# G2.forward = functools.partial(G2.forward, c=None, force_fp32=True)

#G2.eval()
#z = torch.randn([1,1024],dtype=torch.float32).cpu() 
z = torch.from_numpy(np.random.RandomState(0).randn(1, 1024).astype("float32"))
# scripted_model = torch.jit.trace(G2, z)
# torch.onnx.export(
#     scripted_model,
#     (torch.randn((1, 1024), dtype=torch.float32).cpu(), ),
#     "model.onnx",
#     export_params=True,
#     verbose=False,
#     input_names=['input0'],
#     opset_version=10
# )



# # with torch.no_grad():
# #     b = G.mapping(z,c=0,truncation_psi=0.5,truncation_cutoff=None)
# #     d = G.synthesis(b,noise_mode='const',force_fp32=True)
# #     #b = G.forward(z,c=0,truncation_psi=0.5,noise_mode="const")

# # print("G")

# with torch.no_grad():
#     c = G2.mapping(z,c=0,truncation_psi=0.5,truncation_cutoff=None)
#     e = G2.synthesis(c,noise_mode='const',force_fp32=True)
#     f = G2.forward(z,c=0,truncation_psi=0.5,noise_mode="const")


# # print(e)
# print()
# print("hiya")
# print(f)

G3 = G2.mapping
G3.eval()

G4 = G2.synthesis
G4.eval()

with torch.no_grad():
    asd = G3(z)
    #print(asd)
    #print(asd.size())

# torch.onnx.export(G3, (torch.randn((1, 1024), dtype=torch.float32)) , "model.onnx", verbose=True, input_names=["rand_array"], output_names=["arrai_boi"],
#                   opset_version=10)

# # jitty = torch.jit.trace(G4, (torch.randn((10, 16, 1024), dtype=torch.float32)))
# # print(jitty)
# torch.onnx.export(G4, (torch.randn((1, 16, 1024), dtype=torch.float32)) , "modelsynth.onnx", verbose=True, input_names=["latent_w"], output_names=["arrai_boi"],
#                 opset_version=9,operator_export_type=torch.onnx.OperatorExportTypes.ONNX
#                 # ,dynamic_axes={
#                 #     # dict value: manually named axes
#                 #     "latent_w": {0: "batch"},
#                 #     # list value: automatic names
#                 #     #"sum": [0],
#                 # }
#                 )

def create_im(tensor_arr):
    img = (tensor_arr + 1) * 255 / 2  # [-1.0, 1.0] -> [0.0, 255.0]
    img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]  # NCWH => NWHC
    PIL.Image.fromarray(img, 'RGB').save('sangli.jpg')

with torch.no_grad():
    s3 = G4(asd)
    print(s3)
    f = G2.forward(z,c=0,truncation_psi=1.0,noise_mode="const")

create_im(f)

# b = G2.forward(z,
#     torch.zeros([1, 0]),
#     torch.randint(2,10,(1,),dtype=torch.float32))
# img = (b + 1) * 255 / 2  # [-1.0, 1.0] -> [0.0, 255.0]
# img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]  # NCWH => NWHC
# PIL.Image.fromarray(img, 'RGB').save('sangli.jpg')
# m_k = {"mbstd_group_size":32,"mbstd_num_channels":4}
# D1 = Discriminator(
#     c_dim=0,
#     img_resolution=1024,
#     img_channels=3,
#     channel_base=16<<10,
#     epilogue_kwargs=m_k,
# ).cpu()


# d_sd = D.state_dict()
# d2_sd = D1.state_dict()

# for k, _ in d2_sd.items():
#     try: d2_sd[k] = d_sd[k]
#     except: pass

# D1.load_state_dict(d2_sd)


# def validate_state_dicts(model_state_dict_1, model_state_dict_2):
#     if len(model_state_dict_1) != len(model_state_dict_2):
#         logging.info(
#             f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
#         )

#         return False

#     # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
#     if next(iter(model_state_dict_1.keys())).startswith("module"):
#         model_state_dict_1 = {
#             k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
#         }

#     if next(iter(model_state_dict_2.keys())).startswith("module"):
#         model_state_dict_2 = {
#             k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
#         }

#     for ((k_1, v_1), (k_2, v_2)) in zip(
#         model_state_dict_1.items(), model_state_dict_2.items()
#     ):
#         if k_1 != k_2:
#             logging.info(f"Key mismatch: {k_1} vs {k_2}")
#             return False
#         # convert both to the same CUDA device
#         if str(v_1.device) != "cuda:0":
#             v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
#         if str(v_2.device) != "cuda:0":
#             v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

#         if not torch.allclose(v_1, v_2):
#             logging.info(f"Tensor mismatch: {v_1} vs {v_2}")
#             return False

# print(validate_state_dicts(g_sd,g2_sd))

# import torch
# torch.onnx.export(G2, (torch.randn((1, 1024), dtype=torch.float32),torch.zeros([1, 0]),torch.randint(0,1,(1,),dtype=torch.float32)) , "alex1net.onnx", verbose=True, input_names=["rand_array","c","trunc"], output_names=["mangi"],
#                   opset_version=10)


                  
# (lit) C:\Users\Aravind\Desktop\stylegan2-ada-pytorch-main\stylegan2-ada-pytorch-main>python fast.py
# C:\Users\Aravind\Desktop\stylegan2-ada-pytorch-main\stylegan2-ada-pytorch-main\training\networks.py:256: 
# TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. 
# We can't record the data flow of Python values, so this value will be treated as a constant in the future. 
# This means that the trace might not generalize to other inputs!
#   if truncation_psi != 1:
# tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          ...,
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.]]], grad_fn=<AddBackward0>)
