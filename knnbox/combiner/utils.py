r""" some utils function used for combiner """

import torch
import torch.nn.functional as F

def calculate_knn_prob(vals, distances, probability_dim, temperature, device,tgt_tokens=None, **kwargs):
    r"""
    How vanilla knn-mt calculates knn probs using retrieved vals and distances.
    """
    # import pdb;pdb.set_trace()
    scaled_dists = - distances / temperature
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    
    B, S, K = vals.size()

    # construct prob
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    return knn_probs



def calculate_combined_prob(knn_prob=None, neural_model_logit=None, lambda_=None, log_probs=None,prev_output_tokens =None,knn_inference_mode="all"):
    r""" 
    How vanilla knn-mt calculate the combining probability.
    """
    # import pdb;pdb.set_trace()
    neural_model_prob = F.softmax(neural_model_logit, dim=-1)
    # import pdb;pdb.set_trace()
    if knn_inference_mode == "mask":
        mask  = prev_output_tokens.ne(3)
        knn_prob[mask, :] = 0 
        combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)
    else:
        combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)

    ## some extra infomation
    extra = {}
    extra["neural_probs"] = neural_model_prob
    extra["unlog_combined_probs"] = combined_probs

    if log_probs:
        combined_probs =  torch.log(combined_probs)
    return combined_probs, extra


def calculate_knn_prob_with_merge_weight(vals=None, distances=None, merge_weights=None, probability_dim=None, temperature=None, device=None, **kwargs):
    r""" 
    when the key-value pair has a merge weight.
    used by greedy-merge knn-mt
    """
    # consider merge weights here
    scaled_dists = - distances / temperature + torch.log(merge_weights.float())
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    
    B, S, K = vals.size()

    # construct prob
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

    return knn_probs