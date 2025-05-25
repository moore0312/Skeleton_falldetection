import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x: (N, T, V, C) or (N, V, T, C)
        N, T, V, C = x.size()
        q = self.query(x).view(N, T, V, self.num_heads, self.head_dim)
        k = self.key(x).view(N, T, V, self.num_heads, self.head_dim)
        v = self.value(x).view(N, T, V, self.num_heads, self.head_dim)

        q = q.permute(0, 3, 1, 2, 4)  # (N, heads, T, V, head_dim)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (N, heads, T, V, V)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (N, heads, T, V, head_dim)

        out = out.permute(0, 2, 3, 1, 4).contiguous()  # (N, T, V, heads, head_dim)
        out = out.view(N, T, V, -1)  # (N, T, V, embed_dim)
        return self.out(out)


class ST_SAGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=14, num_classes=7, seq_len=30, embed_dim=64, num_heads=4):
        super(ST_SAGCN, self).__init__()
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Fall',
                    'Stand up', 'Sit down', 'Falling']


        self.in_channels = in_channels
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # --- Input Encoding ---
        self.joint_encoder = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.vel_encoder = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # --- Semantic Embedding ---
        self.joint_type_embed = nn.Embedding(num_joints, embed_dim)
        self.frame_idx_embed = nn.Embedding(seq_len, embed_dim)

        # --- Projection ---
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

        # --- Learnable Adjacency Matrix ---
        self.gcn_input_proj = nn.Linear(embed_dim, embed_dim)
        self.gcn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # --- Spatial and Temporal Attention ---
        self.spatial_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.temporal_attn = MultiHeadSelfAttention(embed_dim, num_heads)

        # --- Output Classification ---
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x_joint, x_vel):
        # Input: (N, C, T, V)
        N, C, T, V = x_joint.size()

        # --- Encode inputs ---
        f_joint = self.joint_encoder(x_joint)  # (N, embed, T, V)
        f_vel = self.vel_encoder(x_vel)        # (N, embed, T, V)
        f = f_joint + f_vel

        # --- Semantic Embedding ---
        joint_ids = torch.arange(V, device=x_joint.device).unsqueeze(0).repeat(N, 1)  # (N, V)
        joint_embed = self.joint_type_embed(joint_ids).permute(0, 2, 1).unsqueeze(2).repeat(1, 1, T, 1)  # (N, embed, T, V)

        frame_ids = torch.arange(T, device=x_joint.device).unsqueeze(0).repeat(N, 1)  # (N, T)
        frame_embed = self.frame_idx_embed(frame_ids).permute(0, 2, 1).unsqueeze(3).repeat(1, 1, 1, V)  # (N, embed, T, V)

        # --- Concatenate features ---
        feat = torch.cat([f, joint_embed, frame_embed], dim=1)  # (N, 3*embed, T, V)
        feat = self.proj(feat.permute(0, 2, 3, 1))  # (N, T, V, embed)

        # --- Learnable Adjacency Matrix ---
        norm_feat = F.normalize(feat, dim=-1)  # cosine sim
        adj = torch.matmul(norm_feat, norm_feat.transpose(-2, -1))  # (N, T, V, V)
        adj = F.softmax(adj, dim=-1)
        feat = torch.matmul(adj, feat)  # (N, T, V, embed)

        # --- Graph Convolution ---
        feat = self.gcn(feat)

        # --- Spatial Self-Attention ---
        feat_spa = self.spatial_attn(feat) + feat  # (N, T, V, embed)

        # --- Temporal Self-Attention ---
        feat_tmp = feat_spa.permute(0, 2, 1, 3)  # (N, V, T, embed)
        feat_tmp = self.temporal_attn(feat_tmp) + feat_tmp  # (N, V, T, embed)
        feat = feat_tmp.permute(0, 2, 1, 3)  # (N, T, V, embed)

        # --- Final Classification ---
        feat = feat.permute(0, 3, 1, 2)  # (N, embed, T, V)
        pooled = self.pool(feat).squeeze(-1).squeeze(-1)  # (N, embed)
        out = self.fc(pooled)
        return out

class STSAGCN_Wrapper(nn.Module):
    def __init__(self, num_classes=7):
        super(STSAGCN_Wrapper, self).__init__()
        self.model = ST_SAGCN(num_classes=num_classes)

        # 類別名稱顯示
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Fall',
                            'Stand up', 'Sit down', 'Falling']

    def forward(self, x):
        return self.model(*x)  # 解包 (x_joint, x_vel)
