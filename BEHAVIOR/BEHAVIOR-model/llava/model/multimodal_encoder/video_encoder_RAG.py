import torch
import torch.nn as nn
import math 
import random

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize
from torch_scatter import scatter_mean
from .position_encodings import PositionEmbeddingLearnedMLP
# from .pointnet2_utils import FurthestPointSampling
import torch.nn.functional as F

def farthest_point_sampling(points, npoint, return_indices=False):
    """
    Perform Farthest Point Sampling (FPS) on a batch of point features.
    
    Args: 
        points (torch.Tensor): Input point features of shape [B, N, C].
        npoint (int): Number of points to sample.
        return_indices (bool): If True, also return the indices of the sampled points.
    
    Returns:
        If return_indices is False:
            torch.Tensor: Sampled point features of shape [B, npoint, C].
        Otherwise:
            (sampled_points, centroids) where:
              sampled_points: Tensor of shape [B, npoint, C]
              centroids: Tensor of shape [B, npoint] with the indices of sampled points.
    """
    B, N, C = points.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=points.device)
    # Ensure distances is created with the same dtype as points (e.g., BFloat16)
    distances = torch.full((B, N), 1e10, dtype=points.dtype, device=points.device)
    farthest = torch.randint(0, N, (B,), device=points.device)
    batch_indices = torch.arange(B, device=points.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, dim=2)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=1)[1]

    sampled_points = points[batch_indices.unsqueeze(1), centroids]
    if return_indices:
        return sampled_points, centroids
    else:
        return sampled_points
# # Example usage:
# if __name__ == "__main__":
#     # Input: a tensor of shape [5, 11520, 1024]
#     B, N, C = 5, 11520, 1024
#     points = torch.randn(B, N, C)  # random point features for demonstration

#     # Set the candidate budget (number of points to sample)
#     npoint = 1024

#     # Run FPS
#     sampled_points = farthest_point_sampling(points, npoint)
#     print("Sampled points shape:", sampled_points.shape)  # Expected: [5, 1024, 1024]


class SpatialMemory():
    def __init__(self, norm_q, norm_k, norm_v, mem_dropout=None, 
                 long_mem_size=4000, work_mem_size=5, 
                 attn_thresh=5e-4, sim_thresh=0.95, 
                 save_attn=False, num_patches=None):
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_v = norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.top_k = long_mem_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.num_patches = num_patches
        self.init_mem()
    
    def init_mem(self):
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.mem_imgs = None
        self.lm = 0
        self.wm = 0
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat):
        if self.mem_k is None:
            self.mem_k = feat
        else:
            self.mem_k = torch.cat((self.mem_k, feat), dim=1)

        return self.mem_k
    
    def add_mem_v(self, feat):
        if self.mem_v is None:
            self.mem_v = feat
        else:
            self.mem_v = torch.cat((self.mem_v, feat), dim=1)

        return self.mem_v

    def add_mem_c(self, feat):
        if self.mem_c is None:
            self.mem_c = feat
        else:
            self.mem_c = torch.cat((self.mem_c, feat), dim=1)

        return self.mem_c
    
    def add_mem_pts(self, pts_cur):
        if pts_cur is not None:
            if self.mem_pts is None:
                self.mem_pts = pts_cur
            else:
                self.mem_pts = torch.cat((self.mem_pts, pts_cur), dim=1)
    
    def add_mem_img(self, img_cur):
        if img_cur is not None:
            if self.mem_imgs is None:
                self.mem_imgs = img_cur
            else:
                self.mem_imgs = torch.cat((self.mem_imgs, img_cur), dim=1)

    def add_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None):  
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
            
        if self.mem_count is None:
            self.mem_count = torch.zeros_like(feat_k[:, :, :1])
            self.mem_attn = torch.zeros_like(feat_k[:, :, :1])
        else:
            self.mem_count += 1
            self.mem_count = torch.cat((self.mem_count, torch.zeros_like(feat_k[:, :, :1])), dim=1)
            self.mem_attn = torch.cat((self.mem_attn, torch.zeros_like(feat_k[:, :, :1])), dim=1)
        
        self.add_mem_k(feat_k)
        self.add_mem_v(feat_v)
        self.add_mem_pts(pts_cur)
        self.add_mem_img(img_cur)
    
    def check_sim(self, feat_k, thresh=0.7):
        # Do correlation with working memory
        if self.mem_k is None or thresh==1.0:
            return False
        
        wmem_size = self.wm * self.num_patches

        # wm: BS, T, 196, C
        wm = self.mem_k[:, -wmem_size:].reshape(self.mem_k.shape[0], -1, self.num_patches, self.mem_k.shape[-1])

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        wm_norm = F.normalize(wm, p=2, dim=-1)

        corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)

        mean_corr = torch.mean(corr, dim=-1)

        if mean_corr.max() > thresh:
            print('Similarity detected:', mean_corr.max())
            return True
    
        return False

    def add_mem_check(self, feat_k, feat_v, pts_cur=None, img_cur=None):
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]

        if self.check_sim(feat_k, thresh=self.sim_thresh):
            return
        
        self.add_mem(feat_k, feat_v, pts_cur, img_cur)
        self.wm += 1

        if self.wm > self.work_mem_size:
            self.wm -= 1
            if self.long_mem_size == 0:
                self.mem_k = self.mem_k[:, self.num_patches:]
                self.mem_v = self.mem_v[:, self.num_patches:]
                self.mem_count = self.mem_count[:, self.num_patches:]
                self.mem_attn = self.mem_attn[:, self.num_patches:]
                print('Memory pruned:', self.mem_k.shape)
            else:
                self.lm += self.num_patches
        
        if self.lm > self.long_mem_size:
            self.memory_prune()
            self.lm = self.top_k - self.wm * self.num_patches
    
    def memory_read(self, feat, res=True):
        '''
        Params:
            - feat: [bs, p, c]
            - mem_k: [bs, t, p, c]
            - mem_v: [bs, t, p, c]
            - mem_c: [bs, t, p, 1]
        '''
        
        affinity = torch.einsum('bpc,bxc->bpx', self.norm_q(feat), self.norm_k(self.mem_k.reshape(self.mem_k.shape[0], -1, self.mem_k.shape[-1])))
        affinity /= torch.sqrt(torch.tensor(feat.shape[-1]).float())
        
        if self.mem_c is not None:
            affinity = affinity * self.mem_c.view(self.mem_c.shape[0], 1, -1)  
        
        attn = torch.softmax(affinity, dim=-1)

        if self.save_attn:
            if self.attn_vis is None:
                self.attn_vis = attn.reshape(-1)
            else:
                self.attn_vis = torch.cat((self.attn_vis, attn.reshape(-1)), dim=0)
        if self.mem_dropout is not None:
            attn = self.mem_dropout(attn)
        
        if self.attn_thresh > 0:
            attn[attn<self.attn_thresh] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True) 
        
        out = torch.einsum('bpx,bxc->bpc', attn, self.norm_v(self.mem_v.reshape(self.mem_v.shape[0], -1, self.mem_v.shape[-1])))
        
        if res:
            out = out + feat
        
        
        total_attn = torch.sum(attn, dim=-2)
        self.mem_attn += total_attn[..., None]
        
        return out
    
    def memory_prune(self):

        weights = self.mem_attn / self.mem_count
        weights[self.mem_count<self.work_mem_size+5] = 1e8

        num_mem_b = self.mem_k.shape[1]


        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=1)
        top_k_indices_expanded = top_k_indices.expand(-1, -1, self.mem_k.size(-1))


        self.mem_k = torch.gather(self.mem_k, -2, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, -2, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, -2, top_k_indices)
        self.mem_count = torch.gather(self.mem_count, -2, top_k_indices)
 

        if self.mem_pts is not None:
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 256, 3)
            self.mem_pts = torch.gather(self.mem_pts, 1, top_k_indices_expanded)
            self.mem_imgs = torch.gather(self.mem_imgs, 1, top_k_indices_expanded)

        num_mem_a = self.mem_k.shape[1]

        print('Memory pruned:', num_mem_b, '->', num_mem_a)

class PromptEncoder(nn.Module):
    
    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)
    
    def forward(self, clicks):
        # (n, 3)
        pos_embed = self.encode_pe(clicks) #  (N, F)
        return pos_embed

class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = 'fps' #'fps'   #'voxelize'
        self.voxel_size = 0.2 #0.4
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower
        self.mem_q = nn.Linear(1024, 1024)
        self.mem_k = nn.Linear(1024, 1024)
        self.mem_v = nn.Linear(1024, 1024)
        self.max_rooms =  10 - 1
        self.pos_embed_k = nn.Embedding(self.max_rooms, 1024)
        self.pos_embed_v = nn.Embedding(self.max_rooms, 1024)
        self.mem_dropout = nn.Dropout(0.1)
        self.ln_mem_q = nn.LayerNorm(1024, eps=1e-6)
        self.ln_mem_k = nn.LayerNorm(1024, eps=1e-6)
        self.ln_mem_v = nn.LayerNorm(1024, eps=1e-6)
        self.mem_attn = nn.MultiheadAttention(1024, 16, dropout=0.1, batch_first=True) # self.mem_dropout
        self.proj_out = nn.Parameter((1024 ** -0.5) * torch.randn(1024, 1024))
        # sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v, mem_dropout=self.mem_dropout)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return

        self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, features, world_points, depths, poses, intrinsics, lengths=None):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 336, 336), image token features
            - depths: (B, V, H, W), depth images
            - poses: (B, V, 4, 4) pose information
            - instrinsics: (B, V, 4, 4), intriniscs
            - lengths: (B,)  view number of each scene

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        B, V, C, H, W = features.shape
        # assert intrinsics.dim() == 4
        # (B, V, 24, 24, 3)
        # print('image token features:', features.shape) # ([3, 20, 1024, 24, 24])
        # print('image token features flatten:', features.flatten(0, 1).shape) # torch.Size([2, 5, 1024, 336, 336]) -> ([60, 1024, 24, 24])
        train = True             
        if train:
            bs = world_points.shape[0] # len(world_points) #
        else: 
            # print('world_points:', world_points[0].shape) # torch.Size([5, 20, 336, 336, 3])
            bs = world_points.shape[1]
        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], world_points, depths, poses, intrinsics)
        # print('feat_xyz', feat_xyz.shape) # feat_xyz torch.Size([5, 20, 24, 24, 3])
        # (B, V*H*W, C)
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        # print('video_features:', video_features.shape) # torch.Size([5, 11520, 1024])
        video_xyz = feat_xyz.reshape(B, V*H*W, 3)
        if lengths is not None:
            lengths = lengths*H*W

        if self.pooling == 'voxelize':
            # raise NotImplementedError
            p2v = voxelize(feat_xyz, self.voxel_size)  # （B, N)
            print('p2v:', p2v.shape) # torch.Size([2, 11520])
            # print('video_features:', video_features.shape) # t
            print("len(video_features):", len(video_features)) # 5
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(len(video_features))]) # bn, F4
            print('pooled_video_features:', pooled_video_features.shape) # torch.Size([3759, 1024])
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
            print('batch_offset:', batch_offset) #       device='cuda:3', dtype=torch.int32) tensor([ 470, 1318, 2267, 3430, 3879, 4828, 5277, 6440, 7509, 7979],
        elif self.pooling == 'fps':
            npoint = 1024 
            sampled_points, centroids = farthest_point_sampling(video_xyz, npoint, return_indices=True)
            # Gather video features using the sampled indices for each batch.
            pooled_video_features = torch.cat([video_features[b][centroids[b]] for b in range(len(video_features))])
            # Since FPS produces a fixed number of points per batch, the batch offset is simply:
            batch_offset = torch.arange(npoint, (B + 1) * npoint, npoint, device=feat_xyz.device, dtype=torch.int32)
            print('pooled_video_features:', pooled_video_features.shape)  # Expected: [B * npoint, F] e.g. [5120, 1024]
            print('batch_offset:', batch_offset)
        else:
            raise NotImplementedError
        
        ############ memory QKV ####
        per_sample_len = len(batch_offset) // bs
        last_offset = 0
        batch_q = []
        batch_k = []
        batch_v = []
        # bs_offset = []
        idx = 0 
        working_mem_bs = []
        random_longterm_mem = []
        print('bs:', bs)
        print('per_sample_len:', per_sample_len)
        for offset in range(1, bs+1, 1):
            first_batch_offset = batch_offset[last_offset:offset*per_sample_len]
            video_features = []
            for b in first_batch_offset:
                feats = pooled_video_features[idx:b]
                if feats.shape[0] > 2560:
                    indices = torch.randperm(feats.size(0))[:2560]
                    feats = feats[indices]
                idx = b
                video_features.append(feats)  # [(C, 1214), (C, 2321), ...] len(mini_b_2)
            # bs_offset.append(idx)
            
            working_mem = video_features[-1]
            working_mem_size = len(working_mem)
            working_mem_bs.append(working_mem)
            cur_q = self.mem_q(video_features[-1]) #### -2 

            proj_feats_k = []
            proj_feats_v = []
            for room_id, feat in enumerate(video_features[:-1]):
                # Apply the key and value projection layers
                # proj_k = self.mem_k(feat)  # shape: (n_tokens, d_model)
                # proj_v = self.mem_v(feat)
                
                # Generate positional embeddings for this room
                # Create a tensor with the current room id and expand it to each token's position
                # room_id_tensor = torch.tensor(room_id, device=feat.device)
                # pos_emb_k = self.pos_embed_k(room_id_tensor).unsqueeze(0).expand(feat.size(0), -1)
                # pos_emb_v = self.pos_embed_v(room_id_tensor).unsqueeze(0).expand(feat.size(0), -1)
                
                # Add the positional embeddings
                proj_feats_k.append(feat)
                # proj_feats_v.append(proj_v)
            
            # Concatenate features from all past rooms to form the memory banks
            mem_bank_k = torch.cat(proj_feats_k, dim=0)
            # mem_bank_v = torch.cat(proj_feats_v, dim=0)

            # mem_bank_k = torch.cat([self.mem_k(feat) for feat in video_features[:-1]], dim=0)
            # mem_bank_v = torch.cat([self.mem_v(feat) for feat in video_features[:-1]], dim=0)

            batch_q.append(cur_q)
            batch_k.append(mem_bank_k)
            # batch_v.append(mem_bank_v)
            
            Q = cur_q
            K = mem_bank_k
            Q_global, _ = Q.max(dim=0)              # [C]
            Q_norm = F.normalize(Q_global, dim=-1)  # [C]
       
            K_norm = F.normalize(mem_bank_k, dim=1)  # [N, C]

            # cosine similarity
            similarities = torch.matmul(K_norm, Q_norm)   # [N]

            # retrieve most similar
            top_sim, top_idx = torch.max(similarities, dim=0)
            best_candidate = K[top_idx.item()].unsqueeze(0)  # [C]
            # print('best_candidate', best_candidate.shape)
            # random_longterm_mem.append(video_features[random.randint(0, len(video_features)-2)])
            random_longterm_mem.append(best_candidate)
        
        # if self.pooling == 'fps':
        #     batch_q = torch.stack(batch_q, dim=0)
        #     batch_k = torch.stack(batch_k, dim=0)
        #     batch_v = torch.stack(batch_v, dim=0)
        # else: 
        #     batch_q = torch.nn.utils.rnn.pad_sequence(batch_q,
        #                                          batch_first=False).permute(1, 0, 2)  #torch.stack(batch_q, dim=0)
        #     batch_k = torch.nn.utils.rnn.pad_sequence(batch_k,
        #                                             batch_first=False).permute(1, 0, 2)
        #         #torch.stack(batch_k, dim=0)
        #     batch_v = torch.nn.utils.rnn.pad_sequence(batch_v,
                                                    # batch_first=False).permute(1, 0, 2)
            
        # print('nbatch_q:, batch_k, batch_v:', batch_q.shape, batch_k.shape, batch_v.shape)


        # mem_out = self.mem_attn(
        #     self.ln_mem_q(batch_q),
        #     self.ln_mem_k(batch_k),
        #     self.ln_mem_v(batch_v),
        #     attn_mask=None)[0]
        
        # print('mem_out.shape:', mem_out.shape) #torch.Size([2, 2560, 1024])    
        # assert len(working_mem_bs) == bs 
        
        mem_out = random_longterm_mem

        ########### For use work mem 
        new_pooled_video_features = []
        new_batch_offset = []
        working_mem_offset = 0
        for mem, working_mem in zip(mem_out, working_mem_bs):
            new_pooled_video_features.append(torch.cat((mem, working_mem), dim=0))
            working_mem_offset += len(mem) 
            new_batch_offset.append(working_mem_offset)
            working_mem_offset += len(working_mem)
            new_batch_offset.append(working_mem_offset)
        ############## For Cap only 
        # working_mem_offset = 0
        # for mem in mem_out:
        #     new_pooled_video_features.append(mem)
        #     working_mem_offset += len(mem) 
        #     new_batch_offset.append(working_mem_offset)
        
        new_pooled_video_features = torch.cat(new_pooled_video_features, dim=0)
        print('new_pooled_video_features, new_batch_offset:', new_pooled_video_features.shape, new_batch_offset)
        # new_pooled_video_features = new_pooled_video_features.flatten(0, 1)
        # print('new_pooled_video_features:', new_pooled_video_features.shape) # torch.Size([3759, 1024])
        #mem_out = mem_out @ self.proj_out
        # class CrossOutput(nn.Module):
        #     def __init__(self, config):
        #         super().__init__()
        #         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #         self.LayerNorm = LayerNormFp32(config.hidden_size) #, eps=config.layer_norm_eps)
        #         self.dropout = nn.Dropout(0.1) #config.hidden_dropout_prob) add later
        #         #self.mlp = AbstractorMLP(config)

        #     def forward(self, hidden_states, input_tensor):
        #         hidden_states = self.dense(hidden_states)
        #         hidden_states = self.dropout(hidden_states)
        #         hidden_states = self.LayerNorm(hidden_states + input_tensor)
        #         return hidden_states

        return new_pooled_video_features, new_batch_offset
        # return pooled_video_features, batch_offset  # (B, num_token, 1024) or (Bn, 1024)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
