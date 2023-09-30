import torch

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'alpha' in self.kwargs:
            self.alpha = self.kwargs['alpha']
        else:
            self.alpha = None
        self.create_embedding_fn()

    def get_scalar(self, j, L):
        """j \in [0,L-1], L is frequency length, was taken form paper(HFS)"""
        return (1.0-torch.cos(torch.clamp(self.alpha*L-j,0,1)*torch.pi)) / 2.0

    def update_alpha(self, alpha=1000.):
        if self.alpha is None:
            self.alpha = alpha
        else:
            self.alpha = max(self.alpha + alpha, 0.)
            self.alpha = min(self.alpha, 1.)

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, j=torch.log(freq), L=N_freqs, p_fn=p_fn,
                                 freq=freq: self.get_scalar(j, L)*p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, alpha=1, input_dims=3):
    embed_kwargs = {
        'alpha': alpha,
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim
    # def embed(x, eo=embedder_obj): return eo.embed(x)
    # return embed, embedder_obj.out_dim
