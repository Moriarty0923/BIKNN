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
from fairseq.modules import FairseqDropout

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

def _compute_contrastive_scores(output1,output2,mask):
    output1 = output1.type(torch.float32)
    norm_rep1 = output1 / output1.norm(dim=2, keepdim=True)
    norm_rep1 = norm_rep1.masked_fill(mask.unsqueeze(-1), 0)

    output2 = output2.type(torch.float32)
    norm_rep2 = output2 / output2.norm(dim=2, keepdim=True)
    norm_rep2 = norm_rep2.masked_fill(mask.unsqueeze(-1), 0)

    contrastive_scores = torch.matmul(norm_rep1, norm_rep2.transpose(1,2))

    return contrastive_scores

@register_model("con_cmlmc")
class CONCMLMCNATransformerModel(NATransformerModel):
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
        parser.add_argument("--mask-mode",type=str,choices=["full-mask","one-mask","random-mask",\
                                                            "random-mask-with-iteration","full-mask-with-iteration",\
                                                            "full-mask-with-iteration-corr","full-mask-with-iteration-wrong",\
                                                            "full-mask-with-iteration-all"]\
                                                            )
        parser.add_argument("--knn-inference-mode",type=str,default="all",choices=["first","all"])
        ######################################## KNN arguments ####################################################

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        ######################################## CMLMC Modifications ####################################################
        self.selfcorrection = args.selfcorrection
        self.correctingself = False
        self.replacefactor = args.replacefactor
        ######################################## CMLMC Modifications ####################################################

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, update_num=None, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        word_ins_out_list, word_ins_mask_list = [], []

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        if self.selfcorrection != -1:
            if update_num is not None:
                self.correctingself = (update_num > self.selfcorrection)
                
        self.decoder.dropout_module = FairseqDropout(0.3)
        word_ins_out, feature = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out,tgt_tokens=tgt_tokens)
        self.decoder.dropout_module = FairseqDropout(0.2)
        word_ins_out1, feature1 = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out,tgt_tokens=tgt_tokens)
        contrastive_scores = _compute_contrastive_scores(feature,feature1,prev_output_tokens.ne(self.unk))
        word_ins_mask = prev_output_tokens.eq(self.unk)

        word_ins_out_list += [word_ins_out1]
        word_ins_mask_list += [word_ins_mask]

        if not self.correctingself:
            return {
                "word_ins": {
                    "out": word_ins_out_list, "tgt": tgt_tokens,
                    "mask": word_ins_mask_list, "ls": self.args.label_smoothing,
                    "nll_loss": True
                },
                "length": {
                    "out": length_out, "tgt": length_tgt,
                    "factor": self.decoder.length_loss_factor
                },
                "con_loss":{
                "contrastive_scores" : contrastive_scores,
                "contrastive_labels" : word_ins_mask,
                }
            }
        else:
        ######################################## CMLMC Modifications ####################################################


            valid_token_mask = (prev_output_tokens.ne(self.pad) &
                                prev_output_tokens.ne(self.bos) &
                                prev_output_tokens.ne(self.eos))
            revealed_token_mask = (prev_output_tokens.ne(self.pad) &
                                   prev_output_tokens.ne(self.bos) &
                                   prev_output_tokens.ne(self.eos) &
                                   prev_output_tokens.ne(self.unk))
            masked_input_out,f = self.decoder(normalize=False,
                                            prev_output_tokens=tgt_tokens.masked_fill(valid_token_mask, self.unk),
                                            encoder_out=encoder_out)

            if self.correctingself:
                revealed_length = revealed_token_mask.sum(-1).float()
                replace_length = revealed_length * self.replacefactor

                # ############## partial self correction, least confident replacing #############################
                # sample the fully_masked_input's output, pick the one with the highest probability
                masked_input_out_scores, masked_input_out_tokens = F.log_softmax(masked_input_out, -1).max(-1)
                # ############################################################################
                # ############## the following line implements random replacing by re-sampling the scores from U(0, 1)
                # ############## if the next line is commented out, least confident replacing is used
                masked_input_out_scores.uniform_()
                # ############################################################################

                # Fill any non-revealed position with 2 (0 also works, but use 2.0 so it's compatible with random
                # sampling as well), which is higher than any valid loglikelihood
                masked_input_out_scores.masked_fill_(~revealed_token_mask, 2.0)
                # calculate the number of tokens to be replaced for each sentence, from which to learn self-correction

                # sort the fully_masked_input's output based on confidence,
                # generate the replaced token mask on the least confident 15%
                _, replace_rank = masked_input_out_scores.sort(-1)
                replace_token_cutoff = new_arange(replace_rank) < replace_length[:, None].long()
                replace_token_mask = replace_token_cutoff.scatter(1, replace_rank, replace_token_cutoff)

                # replace the corresponding tokens in the noisy input sentence, with the token from the generated output
                replaced_input_tokens = prev_output_tokens.clone()
                replaced_input_tokens[replace_token_mask] = masked_input_out_tokens[replace_token_mask]

                self.decoder.dropout_module = FairseqDropout(0.3)
                replace_input_out, replace_feature = self.decoder(normalize=False,
                                                prev_output_tokens=replaced_input_tokens,
                                                encoder_out=encoder_out)
                self.decoder.dropout_module = FairseqDropout(0.2)
                replace_input_out1, replace_feature1 = self.decoder(normalize=False,
                                                prev_output_tokens=replaced_input_tokens,
                                                encoder_out=encoder_out)
                # import pdb;pdb.set_trace()

                # replace_contrastive_scores = _compute_contrastive_scores(replace_feature,replace_feature1,~replace_token_mask)

                # ############## Adding output for L_corr calculation #############################
                word_ins_out_list += [replace_input_out1]
                word_ins_mask_list += [replace_token_mask]
                # import pdb;pdb.set_trace()
                # ############## Adding output for L_mask calculation #############################
                # word_ins_out_list += [word_ins_out]
                # word_ins_mask_list += [word_ins_mask]

            return {
                "word_ins": {
                    "out": word_ins_out_list, "tgt": tgt_tokens,
                    "mask": word_ins_mask_list, "ls": self.args.label_smoothing,
                    "nll_loss": True
                },
                "length": {
                    "out": length_out, "tgt": length_tgt,
                    "factor": self.decoder.length_loss_factor
                },
                "con_loss":{
                    "contrastive_scores" : contrastive_scores,
                    "contrastive_labels" : word_ins_mask
                }
                # "replace_con_loss":{
                #     "contrastive_scores" : replace_contrastive_scores,
                #     "contrastive_labels" : replace_token_mask
                # }
            }
        ######################################## CMLMC Modifications ####################################################

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # ############### CMLM's original code ##################################################
        # ######## CMLM made the decision of NOT updating revealed token confidences, which is understandable in a way
        # ######## because the model is not trained for these positions, but on the other hand it means it never
        # ######## updates the revealed tokens on purpose...
        # output_masks = output_tokens.eq(self.unk)
        # ################################## CMLMC ##################################################
        # ######## update all valid token positions (other than BOS, EOS, PAD) ####################
        output_masks = output_tokens.ne(self.pad) & output_tokens.ne(self.bos) & output_tokens.ne(self.eos)
        # ################################## CMLMC ##################################################
        # import pdb;pdb.set_trace()
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step
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
            history=history
        )


@register_model_architecture("con_cmlmc", "con_cmlmc")
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


@register_model_architecture("con_cmlmc", "con_cmlmc_wmt_en_de")
def cmlmc_wmt_en_de(args):
    cmlmc_base_architecture(args)

@register_model_architecture("con_cmlmc", "vanilla_knn_mt@con_cmlmc")
def cmlmc_knn_wmt_en_de(args):
    cmlmc_base_architecture(args)
