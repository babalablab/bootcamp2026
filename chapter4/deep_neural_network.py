import torch
import torch.nn as nn

from gelu import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut): 
        super().__init__()
        self.use_shortcut = use_shortcut
        
        # 5層のネットワークを定義
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
        
    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            # ショートカット接続が有効、かつ入力と出力の形状が同じ場合のみ加算
            if self.use_shortcut and x.shape == layer_out.shape: 
                x = x + layer_out
            else:
                x = layer_out
        return x


def print_gradients(model, x):
    """モデルの勾配を表示する関数"""
    model.zero_grad()
    
    output = model(x)
    target = torch.tensor([[0.]]) # 1行1列のターゲット値
    
    # 出力とターゲットの平均二乗誤差を計算
    loss = nn.MSELoss()(output, target) 
    
    # 誤差逆伝播
    loss.backward() 
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 勾配の絶対値の平均
            print(f"{name:.<20} gradient mean: {param.grad.abs().mean().item():.6f}")

def main():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]]) # 1行3列のテンソル
    
    print("ショートカット接続なしのモデル")
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)
    
    print("\nショートカット接続あり")
    torch.manual_seed(123) 
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)

if __name__ == "__main__":
    main()