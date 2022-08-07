import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SelfAttention_common(nn.Module):
    def __init__(self,emb_size, num_attention_heads):
        super(SelfAttention_common,self).__init__()
        if emb_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads    #12
        self.attention_head_size = int(emb_size / num_attention_heads)   #384/12=32
        self.all_head_size = self.num_attention_heads * self.attention_head_size   #12*32=384

        self.query = nn.Linear(emb_size, self.all_head_size)  #384->384  三个不同结果
        self.key = nn.Linear(emb_size, self.all_head_size)  #384->384
        self.value = nn.Linear(emb_size, self.all_head_size)  #384->384

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)   # (16,50)+(12,32)->(16,50,12,32)
        x = x.view(*new_x_shape)     # (16,50,384)->(16,50,12,32)简单变形展开
        return x.permute(0, 2, 1, 3)    #  (16,50,12,32)->(16,12,50,32) 调换位置

    def forward(self,dp_emb, dp_mask):
        mixed_query_layer = self.query(dp_emb)   # (16,50,384)-(16,50,384) 线性层
        mixed_key_layer = self.key(dp_emb)     # (16,50,384)
        mixed_value_layer = self.value(dp_emb)   # (16,50,384)

        query_layer = self.transpose_for_scores(mixed_query_layer)   # (16,50,384)->(16,12,50,32) 简单变形
        key_layer = self.transpose_for_scores(mixed_key_layer)   # (16,50,384)->(16,12,50,32)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # (16,50,384)->(16,12,50,32)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  #(16,12,50,32)x(16,12,32,50)->(16,12,50,50) 视频α
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #除根号32

        attention_scores = attention_scores - dp_mask #(16,12,50,50)+(16,1,1,50) mask中补的部分是-10000，其它0，让补的部分很小

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  #(16,12,50,50)
        attention_probs = self.dropout(attention_probs) #视频α'

        context_layer = torch.matmul(attention_probs, value_layer) #(16,12,50,50)*(16,12,50,32)->(16,12,50,32) 视频b
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #(16,50,12,32) contiguous相当于深拷贝，不影响理解
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #(16,50)+(384,)->(16,50,384)
        context_layer = context_layer.view(*new_context_layer_shape) #(16,50,12,32)->(16,50,384) 简单变形合并
        return context_layer, key_layer, value_layer #(16,50,384)


class Attention_common(nn.Module):
    def __init__(self,emb_size, num_attention_heads):
        super(Attention_common,self).__init__()
        self.selfAttention_common = SelfAttention_common(emb_size, num_attention_heads) #自注意力机制
        self.attentonLayer = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self,dp_emb, dp_mask):
        dp_mask = dp_mask.unsqueeze(1).unsqueeze(2)
        dp_mask = (1.0 - dp_mask) * 100000.0
        selfAttention_output, K, V = self.selfAttention_common(dp_emb, dp_mask) #(16,50,384)
        attentionLayer_output = self.attentonLayer(selfAttention_output)  #(16,50,384)->(16,50,384) 线性层
        attentionLayer_dropout = self.dropout(attentionLayer_output)
        attention_output = self.LayerNorm(attentionLayer_dropout + dp_emb)
        return attention_output, K, V  #(16，50，384)


class Attention_middle(nn.Module):
    def __init__(self, emb_size, num_attention_heads, hidden_size):
        super(Attention_middle, self).__init__()
        self.AttentionCommon = Attention_common(emb_size, num_attention_heads)
        self.hiddenLayer1 = nn.Linear(emb_size, hidden_size)
        self.hiddenLayer2 = nn.Linear(hidden_size, emb_size)
        self.dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(emb_size)

    def forward(self, dp_emb, dp_mask):
        attention_output, K, V = self.AttentionCommon(dp_emb, dp_mask)

        hiddenLayer1_output = F.relu(self.hiddenLayer1(attention_output))  # 线性层 (16,50,384)->(16,50,1536)
        hiddenLayer2_output = self.hiddenLayer2(hiddenLayer1_output)  # 线性层 (16,50,1536)->(16,50,384)
        hiddenLayer3_dropout = self.dropout(hiddenLayer2_output)
        layer_output = self.LayerNorm(hiddenLayer3_dropout + attention_output)
        return layer_output, K, V

