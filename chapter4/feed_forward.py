import torch
import torch.nn as nn

from gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), 
            GELU(), 
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

def main():
    cfg = {"emb_dim": 768} 
    
    batch_size = 2
    seq_len = 3
    dummy_input = torch.randn(batch_size, seq_len, cfg["emb_dim"])
    
    # FeedForwardクラスのインスタンス化
    ffn = FeedForward(cfg)
    
    # 順伝播の実行
    output = ffn(dummy_input)
    
    print("FeedForward のテスト")
    print(f"入力テンソルの形状: {dummy_input.shape}")
    print(f"出力テンソルの形状: {output.shape}")
    print(f"入力と出力の形状が一致していればOK -> 一致しているか：{dummy_input.shape == output.shape}")

if __name__ == "__main__":
    main()