import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES_BY_DATASET = {
    'RAVDESS': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
}


def get_class_names(dataset_name, n_classes):
    names = CLASS_NAMES_BY_DATASET.get(dataset_name, [])
    if len(names) == n_classes:
        return names
    return [f'class_{idx}' for idx in range(n_classes)]


def topk_accuracy(outputs_np, targets_np, k):
    topk_preds = np.argsort(outputs_np, axis=1)[:, -k:]
    correct = [targets_np[i] in topk_preds[i] for i in range(len(targets_np))]
    return float(np.mean(correct)) * 100.0


def compute_classification_metrics(*, logits_np, targets_np, dataset_name):
    predictions_np = logits_np.argmax(axis=1)
    class_names = get_class_names(dataset_name, logits_np.shape[1])

    top1 = accuracy_score(targets_np, predictions_np) * 100.0
    top5 = topk_accuracy(logits_np, targets_np, k=min(5, logits_np.shape[1]))
    f1_macro = f1_score(targets_np, predictions_np, average='macro', zero_division=0) * 100.0
    f1_weighted = f1_score(targets_np, predictions_np, average='weighted', zero_division=0) * 100.0
    precision_weighted = precision_score(targets_np, predictions_np, average='weighted', zero_division=0) * 100.0
    recall_weighted = recall_score(targets_np, predictions_np, average='weighted', zero_division=0) * 100.0

    per_class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        mask = targets_np == idx
        if int(mask.sum()) > 0:
            per_class_accuracy[class_name] = round(float((predictions_np[mask] == idx).mean()) * 100.0, 4)

    classification_report_dict = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    classification_report_str = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    confusion = confusion_matrix(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
    ).tolist()

    return {
        'predictions_np': predictions_np,
        'class_names': class_names,
        'top1_accuracy': round(top1, 4),
        'top5_accuracy': round(top5, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'precision_weighted': round(precision_weighted, 4),
        'recall_weighted': round(recall_weighted, 4),
        'per_class_accuracy': per_class_accuracy,
        'classification_report_dict': classification_report_dict,
        'classification_report_str': classification_report_str,
        'confusion_matrix': confusion,
    }
