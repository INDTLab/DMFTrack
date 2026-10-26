class FeatureFuser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, 2)
        )
    
    def forward(self, old_feat, new_feat):
        weights = F.softmax(self.attn(old_feat), dim=-1)
        return weights[0]*old_feat + weights[1]*new_feat