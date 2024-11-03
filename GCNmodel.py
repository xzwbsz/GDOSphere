
import torch
import torch.nn as nn
from torch_scatter import scatter_add
class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight)) # 随机初始权重矩阵
        # 偏差这里没用到
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True): # GCN1 : input-原始特征X adj-当前邻接矩阵cur_adj
        support = torch.matmul(input, self.weight) # MP(X,W) = W × X  e.g.1 (2160,3) × (3,3) = (2160,3)
        output = torch.matmul(adj, support) # MP(cur_adj,support) = cur_adj × support = cur_adj × W × X（or Z(t)) | e.g. (2160,2160) × (2160,3) = (2160,3)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())

class GraphGNN(nn.Module):
    def __init__(self, in_dim, out_dim, edge_src_target, sample_edge_num):
        super(GraphGNN, self).__init__()
        #self.device = device
        #e_h = 32 # edge_mlp hidden
        e_out = 16 # edge_mlp out
        n_out = out_dim # node_mlp out
        sampler = (torch.rand(sample_edge_num)*edge_src_target.shape[1]).type(torch.long) #采样索引
        self.edge_src_target = edge_src_target[...,sampler] #采样出40962条边进行计算
        self.e_w = torch.rand(edge_src_target.shape[1],1)
        self.edge_weight = nn.Parameter(nn.init.xavier_uniform_(self.e_w))#变权重需要学习
        self.edge_mlp = nn.Sequential(nn.Linear(in_dim, e_out), 
                                   nn.Sigmoid(),
                                   )
        self.node_mlp = nn.Sequential(nn.Linear(e_out, n_out),
                                   nn.Sigmoid(),
                                   )

    def forward(self, x): # (batch_size=32,station_num=2160,attr_num=3)
        # edge_src_target = edge_src_target.to(self.device) # 节点索引 class中的参数都传入设备 (2,edges_num)
        # edge_weight = edge_weight.to(self.device)
        # self.w = self.w.to(self.device)
        # self.b = self.b.to(self.device)

        edge_src, edge_target = self.edge_src_target # {2,edge_num} -> src {1,edges_num} 和 target {1,edges_num}
        node_src = x[:, edge_src] # {batch_size,station_num,feature_num} -> {batch_size,edges_num,feature_num}
        node_target = x[:, edge_target] # {batch_size,station_num,feature_num} -> {batch_size,edges_num,feature_num}

        edge_w = self.edge_weight.unsqueeze(-1)
        edge_w = edge_w[None, :, :].repeat(node_src.size(0), 1, 1)#.to(self.device) # (edges_num,1) -> (32,edges_num,1)
        out = torch.cat([node_src, node_target, edge_w], dim=-1) #在最后一个维度进行累加 -> (32,edges_num,3+3+1=7)
        out = self.edge_mlp(out) # out传入edge_mlp更新边属性(32,edges_num,30) e_h = 30

        # 汇聚入度的边特征 and 刨除出度的边特征 最后得到本节点的特征
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1)) # For higher version of PyG.
        out = out_add + out_sub
        out = self.node_mlp(out) # 将out传入node_mlp
        return out

class massage_passing_GNN(nn.Module):
    def __init__(self, in_dim, out_dim, edge_src_target):
        super(GraphGNN, self).__init__()
        #self.device = device
        #e_h = 32 # edge_mlp hidden
        e_out = 16 # edge_mlp out
        n_out = out_dim # node_mlp out
        self.edge_src_target = edge_src_target
        self.e_w = torch.rand(edge_src_target.shape[1],1)
        self.edge_weight = nn.Parameter(nn.init.xavier_uniform_(self.e_w))#变权重需要学习
        self.edge_mlp = nn.Sequential(nn.Linear(in_dim, e_out), 
                                   nn.Sigmoid(),
                                   )
        self.node_mlp = nn.Sequential(nn.Linear(e_out, n_out),
                                   nn.Sigmoid(),
                                   )

    def forward(self, x): # (batch_size=32,station_num=2160,attr_num=3)
        # edge_src_target = edge_src_target.to(self.device) # 节点索引 class中的参数都传入设备 (2,edges_num)
        # edge_weight = edge_weight.to(self.device)
        # self.w = self.w.to(self.device)
        # self.b = self.b.to(self.device)

        edge_src, edge_target = self.edge_src_target # {2,edge_num} -> src {1,edges_num} 和 target {1,edges_num}
        node_src = x[:, edge_src] # {batch_size,station_num,feature_num} -> {batch_size,edges_num,feature_num}
        node_target = x[:, edge_target] # {batch_size,station_num,feature_num} -> {batch_size,edges_num,feature_num}

        edge_w = self.edge_weight.unsqueeze(-1)
        edge_w = edge_w[None, :, :].repeat(node_src.size(0), 1, 1)#.to(self.device) # (edges_num,1) -> (32,edges_num,1)
        out = torch.cat([node_src, node_target, edge_w], dim=-1) #在最后一个维度进行累加 -> (32,edges_num,3+3+1=7)
        out = self.edge_mlp(out) # out传入edge_mlp更新边属性(32,edges_num,30) e_h = 30

        # 汇聚入度的边特征 and 刨除出度的边特征 最后得到本节点的特征
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1)) # For higher version of PyG.
        out = out_add + out_sub
        out = self.node_mlp(out) # 将out传入node_mlp
        return out

class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False): 
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size)) # 创建weight
        self.init_params()                                                # 初始化参数

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)   # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class Classification(nn.Module):                                         # 把GraphSAGE的输出链接全连接层每个节点映射到7维
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(nn.Linear(emb_size, num_classes))      
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists 

class GraphSage(nn.Module):

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size                                  # 输入尺寸   1433
        self.out_size = out_size                                      # 输出尺寸   128
        self.num_layers = num_layers                                  # 聚合层数   2
        self.gcn = gcn                                                # 是否使用GCN
        #self.device = device                                          # 使用训练设备
        self.agg_func = agg_func                                      # 聚合函数
        self.raw_features = raw_features                              # 节点特征
        self.adj_lists = adj_lists                                    # 边
        
        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size       # 如果index==1，这中间特征为1433，如果！=1。则特征数为128。
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))  

    def forward(self, nodes_batch):
        lower_layer_nodes = list(nodes_batch)                          # 把当前训练的节点转换成list
        # [527, 1681, 439, 2007, 1439, 963, 699, 131, 1003, 1, 658, 1660, 16, 716, 245, 2577, 501, 1582, 1081, 944]
        nodes_batch_layers = [(lower_layer_nodes,)]                    # 放入的训练节点
        # [([527, 1681, 439, 2007, 1439, 963, 699, 131, 1003, 1, 658, 1660, 16, 716, 245, 2577, 501, 1582, 1081, 944],)]
        for i in range(self.num_layers):                               # 遍历每一次聚合，获得neighbors
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)  
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
                                     # batch涉及到的所有节点，本身+邻居 ，      节点编号->当前字典中顺序index    
            #[([涉及到的所有节点],[{邻居+自己},{邻居+自己}],{节点index}),([batch节点]),] 
        assert len(nodes_batch_layers) == self.num_layers + 1
        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]                           # 聚合自己和邻居的节点
            pre_neighs = nodes_batch_layers[index-1]                    # 涉及到的所有节点，自己和邻居节点，邻居节点编号->字典中编号
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)   # 聚合函数。聚合的节点， 节点特征，集合节点邻居信息
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)   # 第一层的batch节点，没有进行转换
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats)  
                                                                        # 进入SageLayer。weight*concat(node,neighbors)
            pre_hidden_embs = cur_hidden_embs
        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]                    # 记录将上一层的节点编号。
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]       # self.adj_lists边矩阵，获取节点的邻居
        if not num_sample is None:                                      # 对邻居节点进行采样，如果大于邻居数据，则进行采样
            _sample = random.sample                                     # 节点长度小于10 
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # 加入本身节点
        _unique_nodes_list = list(set.union(*samp_neighs))               # 这个batch涉及到的所有节点
        i = list(range(len(_unique_nodes_list)))                         # 建立编号
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))            # 节点编号->当前字典中顺序index
        return samp_neighs, unique_nodes, _unique_nodes_list             # 聚合自己和邻居节点，点的dict，batch涉及到的所有节点

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs        # batch涉及到的所有节点,本身+邻居,邻居节点编号->字典中编号  
        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]  # 是否包含本身
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]  # 在把中心节点去掉
        if len(pre_hidden_embs) == len(unique_nodes):                     # 保留需要使用的节点特征。
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]                                               
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))           # (本层节点数量，邻居节点数量)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # 保存列 每一行对应的邻居真实index做为列。
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]# 保存行 每行邻居数
        mask[row_indices, column_indices] = 1                             # 构建邻接矩阵;
        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)                         # 按行求和，保持和输入一个维度
            mask = mask.div(num_neigh)#.to(embed_matrix.device)            # 归一化操作
            aggregate_feats = mask.mm(embed_matrix)                       # 矩阵相乘，相当于聚合周围邻接信息求和
        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)
        return aggregate_feats