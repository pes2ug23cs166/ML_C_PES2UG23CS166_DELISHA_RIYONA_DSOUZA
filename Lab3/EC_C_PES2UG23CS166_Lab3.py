import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    y = tensor[:, -1]  
    if y.numel() == 0:
        return 0.0

    values, counts = torch.unique(y, return_counts=True)
    probs = counts.float() / counts.sum().float()
    probs = probs[probs > 0]  
    if probs.numel() == 0:
        return 0.0

    entropy = -(probs * torch.log2(probs)).sum()
    return entropy.item()  


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    n_rows, n_cols = tensor.shape
    X_attr = tensor[:, attribute]
    y = tensor[:, -1]

    values, counts = torch.unique(X_attr, return_counts=True)
    avg_info = torch.tensor(0.0)

    for v, cnt in zip(values, counts):
        mask = (X_attr == v)
        y_sub = y[mask]
        weight = cnt.float() / float(n_rows)
        if y_sub.numel() == 0:
            continue
        vals_sub, counts_sub = torch.unique(y_sub, return_counts=True)
        probs_sub = counts_sub.float() / counts_sub.sum().float()
        probs_sub = probs_sub[probs_sub > 0]
        subset_entropy = -(probs_sub * torch.log2(probs_sub)).sum()
        avg_info += weight * subset_entropy

    return avg_info.item()


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    ig = dataset_entropy - avg_info
    return round(ig, 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    n_rows, n_cols = tensor.shape
    gains = {}
    for attr in range(n_cols - 1): 
        gains[attr] = get_information_gain(tensor, attr)

    max_gain = max(gains.values()) if gains else float("-inf")
    best_attrs = [a for a, g in gains.items() if abs(g - max_gain) < 1e-12]
    selected = min(best_attrs) if best_attrs else None

    return gains, selected
