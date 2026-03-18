import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 

    def forward(self, x): 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) 
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift

# テスト
if __name__ == "__main__":
    # 例として バッチサイズ=2, 系列長(seq_len)=3, 埋め込み次元(emb_dim)=4 の入力を想定
    batch_size = 2
    seq_len = 3
    emb_dim = 4
    
    # ダミーの入力テンソルを作成
    dummy_input = torch.randn(batch_size, seq_len, emb_dim)
    
    # LayerNormのインスタンス化
    layer_norm = LayerNorm(emb_dim)
    
    # 順伝播（Forward）の実行
    output = layer_norm(dummy_input)
    
    print(" Layer Normalization Test ")
    print(f"入力シェイプ: {dummy_input.shape}")
    print(f"出力シェイプ: {output.shape}")
    
    # 確認: 最後の次元(dim=-1)の平均がほぼ0、分散がほぼ1になっているか
    print("\n[正規化後の確認]")
    print("出力の平均 (dim=-1):", output.mean(dim=-1))
    print("出力の分散 (dim=-1):", output.var(dim=-1, unbiased=False))