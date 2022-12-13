# uchile_vision
Vision features for robots at Uchile

To add SiLU activation function to pytorch 1.4:

In a terminal:

cd
cd .local/lib/python2.7/site_packages/torch/nn/modules

Edit activation.py file and add

# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
        
