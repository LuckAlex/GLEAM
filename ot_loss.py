import torch
import ot

def calculate_wasserstein_distance(features1, features2, metric):
    """
    计算两个特征分布之间的Wasserstein距离（最优传输距离）。

    参数:
    - features1: torch.Tensor，第一组特征，形状为 (n_samples1, n_features)
    - features2: torch.Tensor，第二组特征，形状为 (n_samples2, n_features)
    - metric: str，计算距离的度量方式，默认为 'euclidean'
    - numItermax: int，最大迭代次数，默认为 1000

    返回:
    - wasserstein_distance: float，计算得到的Wasserstein距离
    """
    # 计算特征之间的距离矩阵
    if metric == 'euclidean':


        M = ot.dist(features1, features2)
    else:
        M = 1 - (features1 @ features2.T)
    # 定义均匀分布
    n = len(features1)
    m = len(features2)
    a = torch.ones(n, device=features1.device) / n
    b = torch.ones(m, device=features2.device) / m

    # 使用POT库计算Wasserstein距离
    wasserstein_distance = ot.emd2(a, b, M)
    del M



    return wasserstein_distance

# 示例使用
if __name__ == "__main__":
    # 生成示例特征数据（PyTorch张量）
    features1 = torch.rand(15, 2048)  # 100 images with 2048-dimensional features
    # features2 = torch.rand(100, 2048)
    features2 = - features1
    # 计算Wasserstein距离
    distance = calculate_wasserstein_distance(features1, features2)
    print(f"Wasserstein Distance: {distance}")
