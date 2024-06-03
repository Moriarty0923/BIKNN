import torch
import torch.nn.functional as F

from knnbox.combiner.utils import calculate_knn_prob, calculate_combined_prob

class Combiner:
    r"""
    A simple Combiner used by vanilla knn-mt
    """

    def __init__(self, lambda_, temperature, probability_dim,knn_inference_mode):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim
        self.knn_inference_mode = knn_inference_mode

    def get_knn_prob(self, vals=None, distances=None, temperature=None, tgt_tokens=None, device="cuda:0", **kwargs):
        r"""
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature  
        if tgt_tokens is None:
            return calculate_knn_prob(vals=vals,distances=distances,probability_dim=self.probability_dim,
                     temperature=temperature, device=device, **kwargs)
        else:
            return calculate_knn_prob(vals=vals,distances=distances,probability_dim=self.probability_dim,
                        temperature=temperature ,device=device,tgt_tokens=tgt_tokens, **kwargs)

    
    def get_combined_prob(self  , knn_prob=None, neural_model_logit=None, lambda_ = None, log_probs = False ,prev_output_tokens =None):
        r""" 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_

        if prev_output_tokens is None:
            return calculate_combined_prob(knn_prob= knn_prob, neural_model_logit=neural_model_logit, lambda_=lambda_, log_probs=log_probs, knn_inference_mode = self.knn_inference_mode)
        else:
            return calculate_combined_prob( knn_prob, neural_model_logit, lambda_, log_probs,prev_output_tokens=prev_output_tokens, knn_inference_mode = self.knn_inference_mode)

        