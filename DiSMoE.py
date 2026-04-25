import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools



class EmbeddingLayer(nn.Module):
    def __init__(self, in_aqi_features, in_meo_features, features_dim,N_aqi,N_meo):
        super(EmbeddingLayer, self).__init__()
        self.in_aqi_features, self.in_meo_features, self.features_dim = in_aqi_features + 8, in_meo_features + 10, features_dim
        self.aqi_idEmbed, self.aqi_monthEmbed, self.aqi_weekdayEmbed, self.aqi_hourEmbed = nn.Embedding(N_aqi, 2), nn.Embedding(13, 2), nn.Embedding(7, 2), nn.Embedding(24, 2)
        self.meo_windEmbed, self.meo_idEmbed, self.meo_monthEmbed, self.meo_weekdayEmbed, self.meo_hourEmbed = nn.Embedding(9, 2), nn.Embedding(N_meo, 2), nn.Embedding(13, 2), nn.Embedding(7, 2), nn.Embedding(24, 2)
        self.W_xa = nn.Parameter(torch.zeros((self.in_aqi_features, self.features_dim), dtype=torch.float))
        self.W_xm = nn.Parameter(torch.zeros((self.in_meo_features, self.features_dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.W_xa.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_xm.data, gain=1.414)

    def forward(self, aqi_inp, meo_inp, aqi_ex, meo_ex):
        aqi_cat = torch.cat([aqi_inp, self.aqi_idEmbed(aqi_ex[..., 0]), self.aqi_monthEmbed(aqi_ex[..., 1]), self.aqi_weekdayEmbed(aqi_ex[..., 2]), self.aqi_hourEmbed(aqi_ex[..., 3])], dim=-1)
        meo_cat = torch.cat([meo_inp, self.meo_windEmbed(meo_ex[..., 0]), self.meo_idEmbed(meo_ex[..., 1]), self.meo_monthEmbed(meo_ex[..., 2]), self.meo_weekdayEmbed(meo_ex[..., 3]), self.meo_hourEmbed(meo_ex[..., 4])], dim=-1)
        return torch.cat([torch.matmul(aqi_cat, self.W_xa), torch.matmul(meo_cat, self.W_xm)], dim=2)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class LinearModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self,seq_len,pred_len,individual,channels):
        super(LinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = channels


        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1)
        

        return x
    
    
class FrequencyHeterogeneousGraphLayer(nn.Module):
    def __init__(self, features_dim, ffn_hidden_dim, dropout, alpha, context_dim_fft,N_aqi,N_meo):
        super(FrequencyHeterogeneousGraphLayer, self).__init__()
        self.N_aqi = N_aqi
        self.N_meo = N_meo
        self.features_dim, self.ffn_hidden_dim, self.dropout, self.alpha, self.context_dim_fft = features_dim, ffn_hidden_dim, dropout, alpha, context_dim_fft
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(features_dim), nn.LayerNorm(features_dim)
        self.dropout_layer1, self.dropout_layer2 = nn.Dropout(p=self.dropout), nn.Dropout(p=self.dropout)
        self.ffn = nn.Sequential(nn.Linear(features_dim, ffn_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_hidden_dim, features_dim))
        self.att_dim = 2 * (features_dim + context_dim_fft) + 1
        self.a_aa, self.a_am, self.a_ma, self.a_mm = [nn.Parameter(torch.zeros((self.att_dim, 1), dtype=torch.cfloat)) for _ in range(4)]
        for p in [self.a_aa, self.a_am, self.a_ma, self.a_mm]:
            nn.init.xavier_uniform_(p.data.real, gain=1.414); nn.init.xavier_uniform_(p.data.imag, gain=1.414)

    def forward(self, x, adj, adj_norm, context_feat_freq):
        B, N, _ = x.shape
        context_b = context_feat_freq.unsqueeze(0).expand(B, -1, -1)
        heter_feat_freq = torch.cat([x, context_b], dim=2)
        heter_pair = torch.cat([heter_feat_freq.unsqueeze(2).expand(-1, -1, N, -1), heter_feat_freq.unsqueeze(1).expand(-1, N, -1, -1)], dim=3)
        adj_norm_b = adj_norm.unsqueeze(0).unsqueeze(3).expand(B, -1, -1, 1)
        heter_all = torch.cat([heter_pair, adj_norm_b], dim=3)
        e_aa = F.leaky_relu(torch.matmul(heter_all[:, :self.N_aqi, :self.N_aqi, :].real, self.a_aa.real).squeeze(3), self.alpha)
        e_am = F.leaky_relu(torch.matmul(heter_all[:, :self.N_aqi, self.N_aqi:, :].real, self.a_am.real).squeeze(3), self.alpha)
        e_ma = F.leaky_relu(torch.matmul(heter_all[:, self.N_aqi:, :self.N_aqi, :].real, self.a_ma.real).squeeze(3), self.alpha)
        e_mm = F.leaky_relu(torch.matmul(heter_all[:, self.N_aqi:, self.N_aqi:, :].real, self.a_mm.real).squeeze(3), self.alpha)
        e = torch.cat([torch.cat([e_aa, e_am], dim=2), torch.cat([e_ma, e_mm], dim=2)], dim=1)
        attention = F.softmax(torch.where(adj.unsqueeze(0) > 0, e, -1e12), dim=2)
        x_new_real = torch.matmul(attention, x.real)
        x_res1 = x.real + self.dropout_layer1(x_new_real)
        x_norm1 = self.layer_norm1(x_res1)
        final_out_real = self.layer_norm2(x_norm1 + self.dropout_layer2(self.ffn(x_norm1)))
        return torch.complex(final_out_real, x.imag)

class FrequencyHeterogeneousGraph(nn.Module):
    def __init__(self, features_dim, dropout, alpha, ffn_hidden_dim, num_encoder_layers,  context_dim_static,N_aqi,N_meo):
        super(FrequencyHeterogeneousGraph, self).__init__()
        self.encoder_layers = nn.ModuleList([FrequencyHeterogeneousGraphLayer(features_dim, ffn_hidden_dim, dropout, alpha, (context_dim_static // 2 + 1),N_aqi,N_meo) for _ in range(num_encoder_layers)])


    def forward(self, x_time, context_feat, adj, adj_norm):
        context_feat_freq = torch.fft.rfft(context_feat, dim=-1)
        x_freq = torch.fft.rfft(x_time, dim=1)
        encodings_complex = []

        cycled_layers = itertools.cycle(self.encoder_layers)
        for t_f in range(x_freq.size(1)):
            layer = next(cycled_layers)
            x_f_t = x_freq[:, t_f, :, :]
            x_f_t  = layer(x_f_t, adj, adj_norm, context_feat_freq)
            encodings_complex.append(x_f_t)

        x_encoded_complex = torch.stack(encodings_complex, dim=1)


        x_encoded = torch.fft.irfft(x_encoded_complex, n=x_time.size(1), dim=1)

        x_encoded_permuted = x_encoded.permute(0, 2, 1, 3).contiguous()
        B, N, T, F = x_encoded_permuted.shape
        Z = x_encoded_permuted
        return Z 




    

class Expert(nn.Module):
    def __init__(self, d, T_in, T_out, topk, features_dim, dropout=0.5):
        super(Expert, self).__init__()
        self.d, self.T_in, self.T_out, self.topk, self.features_dim = d, T_in, T_out, topk, features_dim



        self.affine_s_c = nn.Sequential(
            nn.Linear(features_dim,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d, features_dim)
        )
        
        self.affine_t_c = nn.Sequential(
            nn.Linear(features_dim,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d, features_dim)
        )

        self.affine_s_t = nn.Sequential(
            nn.Linear(T_in,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d, T_in)
        )
        
        self.affine_t_t = nn.Sequential(
            nn.Linear(T_in,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d,d),
            nn.LayerNorm(d),
            nn.LeakyReLU(0.2),
            nn.Linear(d, T_in)
        )
  
    def forward(self, x_time_sub, Z_sub,  shared_linear):
        B, m, T, F = x_time_sub.shape
        x_freq = torch.fft.rfft(x_time_sub, dim=2)
        amp = x_freq.abs().sum(dim=3)
        k = min(self.topk, x_freq.size(2))
        _, topk_idx = torch.topk(amp, k, dim=2)
        mask = torch.zeros_like(x_freq, dtype=torch.bool, device=x_time_sub.device)



        mask = torch.zeros_like(x_freq, dtype=torch.bool) 

        k = topk_idx.shape[-1]
        F = mask.shape[-1]
        idx_to_scatter = topk_idx.unsqueeze(-1).expand(-1, -1, -1, F)


        mask.scatter_(dim=2, index=idx_to_scatter, value=True)


        non_stat = torch.fft.irfft(x_freq * mask, n=T, dim=2)
        stat = x_time_sub - non_stat


        s_c = self.affine_s_c(Z_sub)
        t_c = self.affine_t_c(Z_sub)

        y_nonstat = non_stat * s_c + t_c

        Z_sub = Z_sub.permute(0,1,3,2)
        y_nonstat = y_nonstat.permute(0,1,3,2)
        s_t = self.affine_s_t(Z_sub)
        t_t = self.affine_t_t(Z_sub)
        y_nonstat = y_nonstat * s_t + t_t

        y_nonstat = y_nonstat.permute(0,1,3,2)
        y_nonstat = y_nonstat + non_stat

        
        stat_perm   = stat.permute(0, 1, 3, 2).contiguous()
        stat_resh    = stat_perm.view(B * m, F, T)
        stat_resh = stat_resh.permute(0,2,1)
        
        stat_linear_out = shared_linear(stat_resh)
        stat_linear_out = stat_linear_out.permute(0,2,1)
        
        stat_processed = stat_linear_out.view(B, m, F, T)
        y_stat = stat_processed.permute(0, 1, 3, 2).contiguous()


        
        
        return y_stat, y_nonstat
        




def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float=1.0):

    xx = x.unsqueeze(1)
    xx2 = x.unsqueeze(0)
    D_xx = ((xx - xx2)**2).sum(2)
    K_xx = torch.exp(-D_xx / (2 * sigma**2))

    yy = y.unsqueeze(1)
    yy2 = y.unsqueeze(0)
    D_yy = ((yy - yy2)**2).sum(2)
    K_yy = torch.exp(-D_yy / (2 * sigma**2))

    xy = x.unsqueeze(1)
    yx = y.unsqueeze(0)
    D_xy = ((xy - yx)**2).sum(2)
    K_xy = torch.exp(-D_xy / (2 * sigma**2))
    
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()




class DiSMoE(nn.Module):
    def __init__(self,
                 in_aqi_features, in_meo_features,
                 num_experts, num_heads,
                 hidden_dim, T_out, T_in,
                 topk=8, dropout=0.5, d=32, alpha=0.2, ffn_hidden_dim=32, num_FHG_layers=1,
                 context_dim_static=22, 

                 mmd_sigma: float = 1.0,
                 N_aqi=35,
                 N_meo=18
                 ):
        super(DiSMoE, self).__init__()
        self.N_aqi = N_aqi
        self.N_meo = N_meo
        self.hidden_dim = hidden_dim
        self.num_experts, self.d, self.T_out, self.T_in, = num_experts, d, T_out, T_in
        self.embedding_layer = EmbeddingLayer(in_aqi_features, in_meo_features, hidden_dim,N_aqi,N_meo)
        self.FHG = FrequencyHeterogeneousGraph(hidden_dim, dropout, alpha, ffn_hidden_dim, num_FHG_layers, context_dim_static, N_aqi,N_meo)
        self.context_proj = nn.Linear(context_dim_static, hidden_dim)
        self.router_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.routing_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_experts * num_heads))

        self.experts = nn.ModuleList([Expert(d, T_in, T_out, topk, hidden_dim, dropout) for _ in range(num_experts)])
        self.out_aqi, self.out_temp, self.out_humi, self.out_wind = [nn.Linear(T_in * 2*hidden_dim, T_out) for _ in range(4)]

 

        self.mmd_sigma = mmd_sigma


        self.shared_linear = LinearModel(seq_len=T_in,pred_len=T_in,individual=False,channels=hidden_dim)



    def forward(self, aqi_inp, meo_inp, aqi_ex, meo_ex, context_feat, adj, adj_norm,
                return_balance_loss=False):
        B, N = aqi_inp.size(0), self.N_aqi + self.N_meo
        
        x_time = self.embedding_layer(aqi_inp, meo_inp, aqi_ex, meo_ex)
        Z = self.FHG(x_time, context_feat, adj, adj_norm)

        x_time_permuted = x_time.permute(0, 2, 1, 3)
        mean_embedding = torch.mean(x_time_permuted, dim=2)
        std_embedding = torch.std(x_time_permuted, dim=2)
        context_embedding = self.context_proj(context_feat).unsqueeze(0).expand(B, -1, -1)
        
        attn_input_seq = torch.stack([mean_embedding, std_embedding, context_embedding], dim=2)
        attn_input_flat = attn_input_seq.view(B * N, 3, -1)

        attended_features, _ = self.router_attention(attn_input_flat, attn_input_flat, attn_input_flat)
        router_input = attended_features.mean(dim=1).view(B, N, -1)
        
        rout_stat = router_input.mean(dim=0)                 
        g_raw     = self.routing_net(rout_stat)
        g_stat    = g_raw.view(N, self.num_experts, -1).mean(dim=2)

        
        balance_loss = torch.tensor(0.0,device='cuda')


        static_assign = g_stat.argmax(dim=1)
        if return_balance_loss:
            mask = F.one_hot(static_assign, self.num_experts).float()
            avg_pi = g_stat.mean(dim=0)
            balance_loss = self.num_experts * (mask.mean(dim=0) * avg_pi).sum()


        y_stat_all, y_nonstat_all = torch.zeros((B, N, self.T_in, self.hidden_dim), device=Z.device), torch.zeros((B, N, self.T_in,self.hidden_dim), device=Z.device)

        prototypes_per_expert = []
  

        for e_id in range(self.num_experts):
            idx = torch.nonzero(static_assign == e_id, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            x_sub = x_time[:, :, idx, :].permute(0,2,1,3)


            mu   = x_sub.mean(dim=(0,2))
            std  = x_sub.std(dim=(0,2))
            proto = torch.cat([mu, std], dim=0)
            prototypes_per_expert.append(proto)



            Z_sub = Z   [:,       idx, : ,:  ]              
            




            y_stat_sub, y_nonstat_sub = self.experts[e_id]( x_sub, Z_sub,self.shared_linear)

            y_stat_all   [:, idx, :, :] = y_stat_sub
            y_nonstat_all[:, idx,  :,:] = y_nonstat_sub



        diversity_loss = torch.tensor(0., device=x_time.device)
        if return_balance_loss and len(prototypes_per_expert) > 1:
            total_mmd = 0.0
            num_pairs = 0
            for p_i, p_j in itertools.combinations(prototypes_per_expert, 2):
                total_mmd += mmd_rbf(p_i, p_j, sigma=self.mmd_sigma)
                num_pairs += 1
            avg_mmd = total_mmd / num_pairs
            diversity_loss = -  avg_mmd



        y_stat_aqi, y_stat_meo = y_stat_all[:, :self.N_aqi], y_stat_all[:, self.N_aqi:]
        y_nonstat_aqi, y_nonstat_meo = y_nonstat_all[:, :self.N_aqi], y_nonstat_all[:, self.N_aqi:]
        

        y_aqi = torch.cat([y_stat_aqi,y_nonstat_aqi],dim=-1)
        y_meo = torch.cat([y_stat_meo,y_nonstat_meo],dim=-1)


        y_aqi = y_aqi.reshape(B,self.N_aqi,-1)
        y_meo = y_meo.reshape(B,self.N_meo,-1)

        nonstat_aqi = self.out_aqi(y_aqi)
        nonstat_temp = self.out_temp(y_meo)
        nonstat_humi = self.out_humi(y_meo)
        nonstat_wind = self.out_wind(y_meo)
        
        out_aqi = nonstat_aqi
        out_temp = nonstat_temp
        out_humi = nonstat_humi
        out_wind = nonstat_wind

        if return_balance_loss:
            return out_aqi, out_temp, out_humi, out_wind, balance_loss ,diversity_loss

        return out_aqi, out_temp, out_humi, out_wind