# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel
from fairseq.utils import new_arange
import torch.nn.functional as F
import torch
from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    disable_model_grad,
    enable_module_grad,
    archs,
)
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

def _random_unmasking(tgt_tokens,output_score,p):
    target_masks = (output_score== 1e5)
    target_score = tgt_tokens.clone().float().uniform_()
    target_score.masked_fill_(target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    target_length = target_length * target_length.clone().uniform_()*0.5
    target_length = target_length + 1  # make sure to mask at least one token.

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    return  target_cutoff.scatter(1, target_rank, target_cutoff)

def _random_mask(target_tokens,p):
    target_masks = (
        target_tokens.ne(1) & target_tokens.ne(0) & target_tokens.ne(2)
    )
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    # target_length = target_length * target_length.clone().uniform_() * p 

    target_length = target_length * p 

    # target_length = target_length + 1  # make sure to mask at least one token.
    target_length.clamp_(min=1)

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    prev_target_tokens = target_tokens.masked_fill(
        target_cutoff.scatter(1, target_rank, target_cutoff), 3
    )
    return prev_target_tokens

@register_model("cmlmc_transformer")
class CMLMNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        ######################################## CMLMC arguments ####################################################
        parser.add_argument('--selfcorrection', type=int, default=-1,
                            help='starting from selfcorrection step, use model to generate tokens for the currently'
                                 'unmasked positions (likely wrong), and teach the model to correct it to the right token,'
                                 'aimed at doing beam-search like correction')
        parser.add_argument("--replacefactor", type=float, default=0.30,
                            help="percentage of ground truth tokens replaced during SelfCorrection or GenerationSampling")
        ######################################## CMLMC arguments ####################################################
        ######################################## KNN arguments ####################################################
        parser.add_argument("--knn-mode", type=str,
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
        parser.add_argument("--mask-mode",type=str,)
        parser.add_argument("--knn-inference-mode",type=str,default="all")
        parser.add_argument("--new-datastore", type=str, metavar="STR", default="_new",
                        help="The directory to save/load robustCombiner")
        parser.add_argument("--random-datastore-iteration-times",type=int,default=5)
        ######################################## KNN arguments ####################################################
        ######################################## train metak   #####################################################
        parser.add_argument("--robust-training-sigma", type=float, default=0.01,
                            help="the noise vector is sampled from a Gaussian distribution with variance sigma^2")
        parser.add_argument("--robust-training-alpha0", type=float, default=1.0,
                            help="alpha0 control the initial value of the perturbation ratio (alpha)")
        parser.add_argument("--robust-training-beta", type=int, default=1000,
                            help="beta control the declining speed of the perturbation ratio (alpha)")
        # ? hyper-params for DC & WP networks
        parser.add_argument("--robust-dc-hidden-size", type=int, default=4,
                            help="the hidden size of DC network")
        parser.add_argument("--robust-wp-hidden-size", type=int, default=32,
                            help="the hidden size of WP network")
        parser.add_argument("--robust-wp-topk", type=int, default=8,
                            help="WP network uses the k highest probabilities of the NMT distribution as input")
        parser.add_argument("--knn-max-k", type=int, metavar="N", default=0,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default=None,
                            help="The directory to save/load robustCombiner")
        parser.add_argument("--del-wrong",default=False)
        ###########################################################################################################





    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        ######################################## CMLMC Modifications ####################################################
        self.selfcorrection = args.selfcorrection
        self.correctingself = False
        self.replacefactor = args.replacefactor
        ######################################## CMLMC Modifications ####################################################
        if hasattr(self.args, "knn_mode") and args.knn_mode == "train_metak":
            disable_model_grad(self)
            enable_module_grad(self, "combiner")

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, update_num=None, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        if self.selfcorrection != -1:
            if update_num is not None:
                self.correctingself = (update_num > self.selfcorrection)
        # import pdb;pdb.set_trace()
        word_ins_out = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out,tgt_tokens=tgt_tokens)
        word_ins_mask = prev_output_tokens.eq(self.unk)

        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "build_datastore" and "iteration" in self.args.mask_mode:
            # import pdb;pdb.set_trace()
            step = 1

            valid_mask = tgt_tokens.ne(self.pad) & tgt_tokens.ne(self.eos) & tgt_tokens.ne(self.bos)
            max_step = min(self.args.random_datastore_iteration_times, valid_mask.sum(-1).clamp_(min=1).min())
            
            while step <= max_step: 
                output_tokens=tgt_tokens.clone()
                output_tokens = _random_mask(
                        tgt_tokens, step/max_step
                    )
                _scores, _tokens = self.decoder(
                    normalize=True,
                    prev_output_tokens=output_tokens,
                    encoder_out=encoder_out,
                    tgt_tokens=tgt_tokens,
                    step=step,
                    iteration_finish=step==max_step,
                ).max(-1)

                step+=1
                
        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "renew_datastore":
            # import pdb;pdb.set_trace()
            _scores, tokens = word_ins_out.max(-1)
            wrong_mask = (tokens!=tgt_tokens) & tgt_tokens.ne(self.pad) & tgt_tokens.ne(self.eos) & tgt_tokens.ne(self.bos)
            if wrong_mask.sum()!=0:
                prev_output_tokens = tgt_tokens.masked_fill(wrong_mask,self.unk)
                word_ins_out = self.decoder(normalize=True, prev_output_tokens=prev_output_tokens,encoder_out=encoder_out,tgt_tokens=tgt_tokens)
                word_ins_mask = prev_output_tokens.eq(self.unk)
                
        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "train_metak":
            return word_ins_out

        
        if not self.correctingself:
            return {
                "word_ins": {
                    "out": word_ins_out, "tgt": tgt_tokens,
                    "mask": word_ins_mask, "ls": self.args.label_smoothing,
                    "nll_loss": True
                },
                "length": {
                    "out": length_out, "tgt": length_tgt,
                    "factor": self.decoder.length_loss_factor
                }
            }
     
        ######################################## CMLMC Modifications ####################################################

    def forward_decoder(self, decoder_out, encoder_out, tgt_tokens, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history


        # 11/8 修改了顺序 不然adaptive会有mask
        if step!=0 and (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )
            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())


        # output_masks = output_tokens.eq(self.unk)
        output_masks = output_tokens.ne(self.pad) & output_tokens.ne(self.bos) & output_tokens.ne(self.eos)

        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
            step=step
        ).max(-1)
        # import pdb;pdb.set_trace()
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

       
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )


@register_model_architecture("cmlmc_transformer", "cmlmc_transformer")
def cmlmc_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlmc_transformer", "cmlmc_transformer_wmt_en_de")
def cmlmc_wmt_en_de(args):
    cmlmc_base_architecture(args)

@register_model_architecture("cmlmc_transformer", "vanilla_knn_mt@cmlmc_knn_wmt_en_de")
def cmlmc_knn_wmt_en_de(args):
    cmlmc_base_architecture(args)

@register_model_architecture("cmlmc_transformer", "cmlmc_transformer_iwslt_en_de")
def cmlmc_iwslt_en_de(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    cmlmc_base_architecture(args)

@register_model_architecture("cmlmc_transformer", "cmlmc_transformer_iwslt_so_tg")
def cmlmc_iwslt_so_tg(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    cmlmc_base_architecture(args)





from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("label_smoothed_cross_entropy_for_cmlmc")
class LabelSmoothedCrossEntropyCriterionForCMLMC(
    LabelSmoothedCrossEntropyCriterion
):
    # ? label_smoothed_cross_entropy_for_robust is a CE-loss that passes target to model, which is required by robust training
    def forward(self, model, sample, reduce=True,update_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        # import pdb;pdb.set_trace()
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if update_num is None:
            net_output = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        else:
            net_output = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, update_num=update_num)
        # import pdb;pdb.set_trace()

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # import pdb;pdb.set_trace()
        
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        # mask = sample["prev_target"].ne(model.decoder.unk)
        mask = sample["target"].eq(model.decoder.pad) | sample["target"].eq(model.decoder.eos) | sample["target"].eq(model.decoder.bos)
        loss, nll_loss = self.label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            mask=mask,
        )
        return loss, nll_loss

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.decoder.get_normalized_probs(net_output, log_probs=True,sample=sample)
        # import pdb;pdb.set_trace()
        target = sample['target']
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def label_smoothed_nll_loss(self,lprobs, target, epsilon, ignore_index=None, reduce=True,mask=None):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        # if ignore_index is not None:
            # # import pdb;pdb.set_trace()
            # pad_mask = target.eq(ignore_index)
            # nll_loss.masked_fill_(pad_mask, 0.0)
            # smooth_loss.masked_fill_(pad_mask, 0.0)
        if mask is not None:
            mask=mask.view(-1).unsqueeze(-1)
            nll_loss.masked_fill_(mask,0.0)
            smooth_loss.masked_fill_(mask,0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss