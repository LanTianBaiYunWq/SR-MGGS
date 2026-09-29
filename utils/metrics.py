"""
评估指标计算
包含BER、TPR@FAR、签名稳定性等指标
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_ber(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    计算比特错误率 (Bit Error Rate)
    
    Args:
        sig1: 签名1
        sig2: 签名2
        
    Returns:
        BER值 (0-1)
    """
    assert len(sig1) == len(sig2), "Signature lengths must match"
    return float(np.mean(sig1 != sig2))


def compute_normalized_correlation(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    计算归一化相关性 (NC)
    
    Args:
        sig1: 签名1
        sig2: 签名2
        
    Returns:
        NC值 (-1到1)
    """
    # 转换为 -1/1 表示
    s1 = sig1.astype(np.float32) * 2 - 1
    s2 = sig2.astype(np.float32) * 2 - 1
    
    return float(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8))


def compute_tpr_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far_threshold: float = 1e-5
) -> float:
    """
    计算特定FAR阈值下的TPR
    
    Args:
        genuine_scores: 真实匹配分数 (越高越匹配)
        impostor_scores: 冒充匹配分数
        far_threshold: FAR阈值
        
    Returns:
        TPR值
    """
    # 合并分数和标签
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 找到FAR <= far_threshold的点
    valid_idx = np.where(fpr <= far_threshold)[0]
    if len(valid_idx) == 0:
        return 0.0
    
    return float(tpr[valid_idx[-1]])


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray
) -> Tuple[float, float]:
    """
    计算等错误率 (Equal Error Rate)
    
    Args:
        genuine_scores: 真实匹配分数
        impostor_scores: 冒充匹配分数
        
    Returns:
        (EER, 阈值)
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # 找到FPR和FNR相等的点
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return float(eer), float(thresholds[eer_idx])


def compute_auc(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray
) -> float:
    """
    计算AUC值
    
    Args:
        genuine_scores: 真实匹配分数
        impostor_scores: 冒充匹配分数
        
    Returns:
        AUC值
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


class SignatureEvaluator:
    """
    签名评估器
    计算各种评估指标
    """
    
    def __init__(self, signature_bits: int = 256):
        """
        Args:
            signature_bits: 签名位数
        """
        self.signature_bits = signature_bits
        self.reset()
    
    def reset(self):
        """重置评估器状态"""
        self.genuine_bers = []      # 同源签名BER
        self.impostor_bers = []     # 异源签名BER
        self.perturbation_results = {}  # 扰动测试结果
    
    def add_genuine_pair(self, sig1: np.ndarray, sig2: np.ndarray):
        """添加同源签名对"""
        ber = compute_ber(sig1, sig2)
        self.genuine_bers.append(ber)
    
    def add_impostor_pair(self, sig1: np.ndarray, sig2: np.ndarray):
        """添加异源签名对"""
        ber = compute_ber(sig1, sig2)
        self.impostor_bers.append(ber)
    
    def add_perturbation_result(
        self,
        perturbation_type: str,
        param: float,
        original_sig: np.ndarray,
        perturbed_sig: np.ndarray
    ):
        """添加扰动测试结果"""
        if perturbation_type not in self.perturbation_results:
            self.perturbation_results[perturbation_type] = {}
        
        if param not in self.perturbation_results[perturbation_type]:
            self.perturbation_results[perturbation_type][param] = []
        
        ber = compute_ber(original_sig, perturbed_sig)
        self.perturbation_results[perturbation_type][param].append(ber)
    
    def compute_metrics(self, far_threshold: float = 1e-5) -> Dict:
        """
        计算所有评估指标
        
        Args:
            far_threshold: FAR阈值
            
        Returns:
            指标字典
        """
        metrics = {}
        
        genuine_bers = np.array(self.genuine_bers)
        impostor_bers = np.array(self.impostor_bers)
        
        # 基本统计
        if len(genuine_bers) > 0:
            metrics['genuine_ber_mean'] = float(np.mean(genuine_bers))
            metrics['genuine_ber_std'] = float(np.std(genuine_bers))
            metrics['genuine_ber_max'] = float(np.max(genuine_bers))
        
        if len(impostor_bers) > 0:
            metrics['impostor_ber_mean'] = float(np.mean(impostor_bers))
            metrics['impostor_ber_std'] = float(np.std(impostor_bers))
            metrics['impostor_ber_min'] = float(np.min(impostor_bers))
        
        # 分离度
        if len(genuine_bers) > 0 and len(impostor_bers) > 0:
            # 使用相似度分数（1 - BER）
            genuine_scores = 1 - genuine_bers
            impostor_scores = 1 - impostor_bers
            
            metrics['tpr_at_far'] = compute_tpr_at_far(
                genuine_scores, impostor_scores, far_threshold
            )
            metrics['eer'], metrics['eer_threshold'] = compute_eer(
                genuine_scores, impostor_scores
            )
            metrics['auc'] = compute_auc(genuine_scores, impostor_scores)
            
            # 分布间隔
            metrics['separability'] = float(
                (np.mean(impostor_bers) - np.mean(genuine_bers)) /
                (np.std(genuine_bers) + np.std(impostor_bers) + 1e-8)
            )
        
        # 扰动稳定性
        perturbation_metrics = {}
        for ptype, param_results in self.perturbation_results.items():
            perturbation_metrics[ptype] = {}
            for param, bers in param_results.items():
                perturbation_metrics[ptype][param] = {
                    'mean_ber': float(np.mean(bers)),
                    'std_ber': float(np.std(bers)),
                    'max_ber': float(np.max(bers))
                }
        metrics['perturbation'] = perturbation_metrics
        
        return metrics
    
    def summary(self) -> str:
        """生成评估摘要"""
        metrics = self.compute_metrics()
        
        lines = [
            "=" * 50,
            "Signature Evaluation Summary",
            "=" * 50,
        ]
        
        if 'genuine_ber_mean' in metrics:
            lines.extend([
                f"Genuine BER:  {metrics['genuine_ber_mean']:.4f} ± {metrics['genuine_ber_std']:.4f}",
                f"Impostor BER: {metrics['impostor_ber_mean']:.4f} ± {metrics['impostor_ber_std']:.4f}",
            ])
        
        if 'tpr_at_far' in metrics:
            lines.extend([
                f"TPR@FAR=1e-5: {metrics['tpr_at_far']:.4f}",
                f"EER:          {metrics['eer']:.4f}",
                f"AUC:          {metrics['auc']:.4f}",
                f"Separability: {metrics['separability']:.4f}",
            ])
        
        if metrics.get('perturbation'):
            lines.append("\nPerturbation Robustness:")
            for ptype, params in metrics['perturbation'].items():
                lines.append(f"  {ptype}:")
                for param, result in params.items():
                    lines.append(
                        f"    {param}: BER = {result['mean_ber']:.4f} ± {result['std_ber']:.4f}"
                    )
        
        lines.append("=" * 50)
        return "\n".join(lines)


class EmbeddingEvaluator:
    """
    嵌入空间评估器
    评估嵌入的质量
    """
    
    def __init__(self):
        self.embeddings = []
        self.labels = []
    
    def add(self, embedding: np.ndarray, label: int):
        """添加嵌入和标签"""
        self.embeddings.append(embedding)
        self.labels.append(label)
    
    def compute_metrics(self) -> Dict:
        """计算嵌入质量指标"""
        embeddings = np.array(self.embeddings)
        labels = np.array(self.labels)
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        metrics = {'n_samples': len(embeddings), 'n_classes': n_classes}
        
        if n_classes < 2:
            return metrics
        
        # 计算类内和类间距离
        intra_distances = []
        inter_distances = []
        
        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            other_embeddings = embeddings[labels != label]
            
            # 类内距离
            if len(class_embeddings) > 1:
                for i in range(len(class_embeddings)):
                    for j in range(i + 1, len(class_embeddings)):
                        dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                        intra_distances.append(dist)
            
            # 类间距离
            if len(other_embeddings) > 0:
                for emb in class_embeddings:
                    dists = np.linalg.norm(other_embeddings - emb, axis=1)
                    inter_distances.extend(dists.tolist())
        
        if intra_distances:
            metrics['intra_distance_mean'] = float(np.mean(intra_distances))
            metrics['intra_distance_std'] = float(np.std(intra_distances))
        
        if inter_distances:
            metrics['inter_distance_mean'] = float(np.mean(inter_distances))
            metrics['inter_distance_std'] = float(np.std(inter_distances))
        
        # 分离度
        if intra_distances and inter_distances:
            metrics['distance_ratio'] = (
                np.mean(inter_distances) / (np.mean(intra_distances) + 1e-8)
            )
        
        return metrics

