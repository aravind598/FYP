# import onnxruntime as ort
# import PIL
# import torch
# import numpy as np
# ort_session = ort.InferenceSession(r"C:\Users\Aravind\Desktop\stylegan2-ada-pytorch-main\stylegan2-ada-pytorch-main\alex1net.onnx",providers=['CPUExecutionProvider'])
# z = np.random.RandomState(0).randn(1, 1024).astype("float32")
# outputs = ort_session.run(
#     None,
#     {"rand_array": z.astype(np.float32),
#     "c":np.asarray([0],dtype=np.float32),
#     "trunc":torch.randint(0,1,(1,),dtype=torch.float32).numpy()
#     }
# )
# img = (torch.from_numpy(outputs[0]).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'seed697.png')
import torch

a = torch.zeros([1, 1])
b = 3
c = a.mul(b)
print(c,c.size())