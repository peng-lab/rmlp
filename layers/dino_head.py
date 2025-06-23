import torch

def _build_mlp(inp_dim=384,
               out_dim=65536,
               hidden_dim=2048,
               bottleneck_dim=256,
               amplitude=1e2,
               rmlp=True):

    if rmlp:
        return RMLP(inp_dim=inp_dim,
                    out_dim=out_dim,
                    amplitude=amplitude,
                    hidden_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim)
    else:
        return MLP(inp_dim=inp_dim,
                    out_dim=out_dim,
                    amplitude=amplitude,
                    hidden_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim)


class RMLP:
    """
    Randomized-Multi-Layer Perceptron used for regularization during contrqstive learning
    """

    def __init__(self, inp_dim: int = 384,
                 out_dim: int = 65536,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256,
                 amplitude: float = 5):
        self.inp_classes = inp_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.amplitude = amplitude

    def __call__(self, x: torch.Tensor):
        w = torch.matmul(x, (torch.eye(self.inp_classes, self.hidden_dim) + torch.normal(0,
                                                                                         self.amplitude / self.hidden_dim,
                                                                                         (self.inp_classes,
                                                                                          self.hidden_dim))).to(
            x.device).type(x.dtype))
        w = torch.nn.GELU()(w)
        w = torch.matmul(w, (torch.eye(self.hidden_dim, self.hidden_dim) + torch.normal(0,
                                                                                        self.amplitude / self.hidden_dim,
                                                                                        (self.hidden_dim,
                                                                                         self.hidden_dim))).to(
            x.device).type(x.dtype))
        w = torch.nn.GELU()(w)
        w = torch.matmul(w, (torch.eye(self.hidden_dim, self.bottleneck_dim) + torch.normal(0,
                                                                                            self.amplitude / self.bottleneck_dim,
                                                                                            (self.hidden_dim,
                                                                                             self.bottleneck_dim))).to(
            x.device).type(x.dtype))
        w = torch.nn.GELU()(w)
        w = torch.matmul(w, (torch.eye(self.bottleneck_dim, self.out_dim) + torch.normal(0,
                                                                                         self.amplitude / self.out_dim,
                                                                                         (self.bottleneck_dim,
                                                                                          self.out_dim))).to(
            x.device).type(x.dtype))
        return w


class MLP(torch.nn.Module):
    def __init__(self, inp_dim=384, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.linear_1 = torch.nn.Linear(inp_dim,hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.linear_3 = torch.nn.Linear(hidden_dim,bottleneck_dim)
        self.linear_4 = torch.nn.Linear(bottleneck_dim,out_dim)

    def forward(self,x):
        x = torch.nn.GELU()(self.linear_1(x))
        x = torch.nn.GELU()(self.linear_2(x))
        x = torch.nn.GELU()(self.linear_3(x))
        return self.linear_4(x)
