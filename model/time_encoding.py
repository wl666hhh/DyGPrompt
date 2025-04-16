import torch
import torch.nn as nn
import numpy as np
class TimeEncoder(torch.nn.Module):
  """
  输入:[batch , ]
  输出:[batch , time_dimension]
  """

  def __init__(self, time_dimension,device='cpu'):
    super(TimeEncoder, self).__init__()
    assert time_dimension%2==0,"时间戳的输出维度应该为偶数d"
    self.time_dimension = time_dimension
    # 初始化频率参数 w1, w2, ..., wd/2
    self.w = nn.Parameter(torch.randn(time_dimension // 2).unsqueeze(0)).to(device)

  def forward(self, t,device='cpu'):
    #判断输入是否为tensor
    if isinstance(t, list):
      t = torch.tensor(t)
    t=t.to(device)
    t.unsqueeze_(-1)
    # 计算时间 t 对应的正弦和余弦编码
    sin_cos_t =torch.zeros((t.shape[0], self.time_dimension)).to(device)
    cos_t=torch.cos(torch.matmul(t, self.w))
    sin_t = torch.sin(torch.matmul(t, self.w))
    sin_cos_t[:, ::2] = cos_t
    sin_cos_t[:, 1::2] =sin_t
    # 返回拼接后的时间编码 [batch,time_dimension]
    return sin_cos_t

if __name__=='__main__':
  time_encoder = TimeEncoder(time_dimension=16,device='cuda:0')  # 设置时间编码的维度为16
  t = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 示例时间戳 [batch(节点个数),]
  encoded_time = time_encoder(t,device='cuda:0')
  print(encoded_time)