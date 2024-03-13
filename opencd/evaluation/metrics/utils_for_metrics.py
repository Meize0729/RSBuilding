def compute_metrics_tools(pred, gt):
    tp = ((pred == 1) & (gt == 1)).float().sum()  # 真阳性：预测为1且实际为1
    fp = ((pred == 1) & (gt == 0)).float().sum()  # 假阳性：预测为1但实际为0
    fn = ((pred == 0) & (gt == 1)).float().sum()  # 假阴性：预测为0但实际为1
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9) # 避免除以0
    return precision.item(), recall.item(), iou.item(), f1_score.item()
