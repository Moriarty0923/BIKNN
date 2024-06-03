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
from fairseq.models.nat import NATransformerModel,FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.utils import new_arange

from fairseq.modules import FairseqDropout

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from fairseq.models.transformer import Embedding
from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner
import torch.nn.functional as F 

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)

def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t

@register_model("con_cmlm")
class CONCMLMNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
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
        parser.add_argument("--mask-mode",type=str,choices=["full-mask","one-mask","random-mask",\
                                                            "random-mask-with-iteration","full-mask-with-iteration",\
                                                            "full-mask-with-iteration_corr","full-mask-with-iteration_wrong"]\
                                                            )
        parser.add_argument("--knn-inference-mode",type=str,default="all",choices=["first","all"])


    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        if src_lengths[0]>10:
            pass
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        # import pdb;pdb.set_trace()
        self.decoder.dropout_module = FairseqDropout(0.3)
        output_layer1, output1 = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
        )
        self.decoder.dropout_module = FairseqDropout(0.2)

        output_layer2, output2 = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
        )
        # import pdb;pdb.set_trace()
        mask = prev_output_tokens.ne(self.unk)
        output1 = output1.type(torch.float32)
        norm_rep1 = output1 / output1.norm(dim=2, keepdim=True)
        norm_rep1 = norm_rep1.masked_fill(mask.unsqueeze(-1), 0)
  

        output2 = output2.type(torch.float32)
        norm_rep2 = output2 / output2.norm(dim=2, keepdim=True)
        norm_rep2 = norm_rep2.masked_fill(mask.unsqueeze(-1), 0)

        contrastive_scores = torch.matmul(norm_rep1, norm_rep2.transpose(1,2))
        # contrastive_scores = contrastive_scores.masked_fill((contrastive_scores==0), -float("inf"))
        word_ins_mask = prev_output_tokens.eq(self.unk)

        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "build_datastore" and "iteration" in self.args.mask_mode:
            self.decoder.dropout_module = FairseqDropout(0.3)
            output_scores, output_tokens = F.log_softmax(output_layer1).max(-1)
            # output_tokens[~output_masks] = tgt_tokens[~output_masks]
            # output_scores[~output_masks] = 1e3
            step = 1
            max_step = min(10,len(output_tokens[0])-1)
            max_score = torch.full_like(output_scores,1e3)
            # import pdb;pdb.set_trace()
            while step < max_step:
                # if len(output_tokens[0])>10: 
                #     import pdb;pdb.set_trace()
                #     pass
                output_masks = (tgt_tokens != self.pad) 
                skeptical_mask = _skeptical_unmasking(
                    output_scores, output_masks, 1 - step / max_step
                ) 
                if skeptical_mask.float().sum()==0:
                    break
                # import pdb;pdb.set_trace()
                output_tokens.masked_fill_(skeptical_mask, self.unk)
                output_scores.masked_fill_(skeptical_mask, 0.0)
                
                output_masks = skeptical_mask

                # datastore = self.decoder.datastore
                # datastore["vals"].add(tgt_tokens[output_masks])
                # datastore.set_pad_mask(output_masks)

                _score,_tokens = self.decoder(
                    normalize=True,
                    prev_output_tokens=output_tokens,
                    encoder_out=encoder_out,
                    tgt_tokens=tgt_tokens,
                    step=step,
                ).max(-1)
                ############    use tgt_token
                output_tokens.masked_scatter_(output_masks, tgt_tokens[output_masks])
                output_scores.masked_scatter_(output_masks, max_score[output_masks])
                ############    use result
                # output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
                # output_scores.masked_scatter_(output_masks, _score[output_masks])
                step += 1
            
            word_ins_mask  = prev_output_tokens.eq(self.unk)

        return {
            "word_ins_1": {
                "out": output_layer1,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_ins_2": {
                "out": output_layer2,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
            "con_loss":{
                "contrastive_scores" : contrastive_scores,
                "contrastive_labels" : prev_output_tokens.eq(self.unk),
            }
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )



@register_model_architecture("con_cmlm", "con_cmlm")
def cmlm_knn_base_architecture(args):
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


@register_model_architecture("con_cmlm", "con_cmlm_wmt_en_de")
def cmlm_knn_wmt_en_de(args):
    cmlm_knn_base_architecture(args)

@register_model_architecture("con_cmlm", "vanilla_knn_mt@con_cmlm")
def cmlm_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    cmlm_knn_base_architecture(args)

@register_model_architecture("con_cmlm", "con_cmlm_wmt_en_de_big")
def cmlm_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    cmlm_knn_base_architecture(args)

@register_model_architecture("con_cmlm", "con_cmlm_8192")
def cmlm_wmt_en_de_8192(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    cmlm_wmt_en_de_big(args)