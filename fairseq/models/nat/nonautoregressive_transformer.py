# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs,\
    select_correct_position_mask,select_wrong_position_mask,select_all_position_mask
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner,RobustCombiner

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


@register_model("nonautoregressive_transformer")
class NATransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
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
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
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

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens,tgt_tokens=None):
        # length prediction
        # 11/9 测试加上 tgt_tokens,后面删
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens
        )
        # import pdb;pdb.set_trace()
        #  11/8 最小长度改成3吧
        #  11/14 设置了最长长度为src_len的1.5  ### 11/15 去掉限制
        # src_len = torch.floor(torch.sum((src_tokens!=self.pad),dim=1)*1.5).type_as(length_tgt)
        # length_tgt.clamp_(max=src_len)
        max_length = length_tgt.clamp_(min=3).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size,src_lengths=None):
        # import pdb;pdb.set_trace()
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        #  11/8 最小长度改成3吧 
        #  11/15 设置一个beam为src_length 
        # import pdb;pdb.set_trace()
        length_tgt = length_tgt.view(-1).clamp_(min=3)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)
        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)
        self.insertCausalSelfAttn = args.insertCausalSelfAttn

        self.update_num = 0
        # import pdb;pdb.set_trace()
        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif hasattr(self.args, "knn_mode") and self.args.knn_mode == "inference":
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["keys","vals"])
            self.datastore.load_faiss_index(["keys"])
            if hasattr(self.args, "knn_max_k") and self.args.knn_max_k!=0 :
                self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
                self.combiner = RobustCombiner.load(args.knn_combiner_path)
            else:
                self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
                self.combiner = Combiner(lambda_=args.knn_lambda,
                        temperature=args.knn_temperature, probability_dim=len(dictionary),knn_inference_mode=args.knn_inference_mode)

        elif hasattr(self.args, "knn_mode") and self.args.knn_mode == "renew_datastore":
            if "datastore" not in global_vars():
                global_vars()['datastore'] = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
            self.datastore = global_vars()['datastore'] 
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                    temperature=args.knn_temperature, probability_dim=len(dictionary),knn_inference_mode=args.knn_inference_mode)
            if "corr_times" not in global_vars():
                global_vars()["corr_times"] = torch.zeros_like(torch.tensor(self.datastore["vals"].data)).to("cuda")
                global_vars()["wrong_times"] = torch.zeros_like(torch.tensor(self.datastore["vals"].data)).to("cuda")
            self.corr_times =   global_vars()["corr_times"]
            self.wrong_times =  global_vars()["wrong_times"]
            if "new_datastore" not in global_vars():
                global_vars()["new_datastore"] = Datastore(args.knn_datastore_path+args.new_datastore)  
            self.new_datastore = global_vars()["new_datastore"]

        
        elif hasattr(self.args, "knn_mode") and self.args.knn_mode == "train_metak":
            if "datastore" not in global_vars():
                global_vars()['datastore'] = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
            self.datastore = global_vars()['datastore'] 
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            self.combiner = RobustCombiner(
                max_k=args.knn_max_k, 
                midsize=args.robust_wp_hidden_size, 
                midsize_dc=args.robust_dc_hidden_size, 
                topk_wp=args.robust_wp_topk, 
                probability_dim=len(dictionary)
            )

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, tgt_tokens=None, step=0, iteration_finish=False, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )

       ################  knn code #####################
        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "build_datastore":
            ####  origin  ####
            # keys = select_keys_with_pad_mask(features, self.datastore.get_pad_mask())
            # self.datastore["keys"].add(keys.half())
            _score,_tokens = self.output_layer(features).max(-1)
            unk_mask = prev_output_tokens.eq(self.unk)
            token_mask = tgt_tokens.ne(self.pad)  & tgt_tokens.ne(self.bos) & tgt_tokens.ne(self.eos)
            mask=None
            keys=None
            if "withoutfirst" in self.args.mask_mode and step == 0:
                pass
            elif "corr" in self.args.mask_mode:
                mask, keys = select_correct_position_mask(features,_tokens,tgt_tokens,token_mask)
            elif "wrong" in self.args.mask_mode:
                mask, keys = select_wrong_position_mask(features,_tokens,tgt_tokens,token_mask)
            elif "all" in self.args.mask_mode:
                ### tgt --> prev 11.8   ### add tgt.ne(self.pad) 11.13 
                mask = prev_output_tokens.ne(self.pad) & prev_output_tokens.ne(self.bos) \
                    & prev_output_tokens.ne(self.eos)  
                keys = select_all_position_mask(features,_tokens,mask)
            else:
                mask = unk_mask
                keys = select_keys_with_pad_mask(features,unk_mask)
    
            vals = tgt_tokens[mask]
 
            if (keys==None) or (vals==None):
                pass

            else:
                self.datastore["keys"].add(keys.half())
                self.datastore["vals"].add(vals)
                # import pdb;pdb.set_trace()
                # self.datastore.tmp_keys.append(keys)
                # self.datastore.tmp_vals.append(vals)

                # if iteration_finish:
                #     keys = self.datastore.tmp_keys
                #     values = self.datastore.tmp_vals
                #     def compute_cosine_similarity(vectors):
                #         # 计算余弦相似度
                #         similarity_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
                #         return similarity_matrix
                #     def generate_mask_for_high_similarity(tensor_list, threshold=0.9):
                #         # 将列表中的张量展开并合并成一个大张量
                #         combined_tensor = torch.stack(tensor_list)  # 形状为 [batch_size, 8, 512]
                #         batch_size, num_positions, feature_dim = combined_tensor.shape
                        
                #         # 初始化掩码，全为1
                #         mask = torch.ones_like(combined_tensor, dtype=torch.bool)
                        
                #         for pos in range(num_positions):
                #             # 获取所有张量在当前 position 的向量
                #             vectors = combined_tensor[:, pos, :]  # 形状为 [batch_size, 512]
                #             similarity_matrix = compute_cosine_similarity(vectors)
                            
                #             for i in range(batch_size):
                #                 if mask[i, pos].all():
                #                     # 找到与当前向量相似度高于阈值的向量
                #                     high_sim_indices = torch.where(similarity_matrix[i] > threshold)[0]
                #                     # 将这些向量标记为 False，除了当前向量
                #                     high_sim_indices = high_sim_indices[high_sim_indices != i]
                #                     mask[high_sim_indices, pos] = False
                        
                #         return mask
                #     mask = generate_mask_for_high_similarity(keys, threshold=0.9)



                #     self.datastore["keys"].add(keys.half())
                #     self.datastore["vals"].add(vals)
                #     self.datastore.tmp_keys = []
                #     self.datastore.tmp_vals = []
              
        
        elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["inference", "train_metak", "renew_datastore"]:
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            # 
            self.retriever.retrieve(features, return_list=["vals", "keys", "query", "distances","indices"])
        elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["renew_datastore"]:

            self.retriever.retrieve(features, return_list=["vals","keys", "query", "distances","indices"])
        extra.update({"last_hidden": features, "target": tgt_tokens})

        decoder_out = self.output_layer(features)

        if normalize:
            if hasattr(self.args, "knn_mode") and self.args.knn_mode == "inference":
                if not hasattr(self.args, "knn_max_k") or self.args.knn_max_k==0:
                    knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=decoder_out.device)
                    combined_prob, _ = self.combiner.get_combined_prob( knn_prob=knn_prob, neural_model_logit=decoder_out, log_probs=True)
                else:
                    combined_prob = self.get_normalized_probs((decoder_out,extra), log_probs=True, prev_output_tokens = prev_output_tokens)
                return combined_prob
            
            if hasattr(self.args, "knn_mode") and self.args.knn_mode == "renew_datastore":
                mask = prev_output_tokens.eq(self.unk)
                if mask.sum()==0:
                    return decoder_out
                keys = select_keys_with_pad_mask(features,mask)
                vals = tgt_tokens[mask]
                self.new_datastore["keys"].add(keys.half())
                self.new_datastore["vals"].add(vals)
                return decoder_out
          
            else:
                return F.log_softmax(decoder_out, -1)
        else:
            if hasattr(self.args, "knn_mode") and self.args.knn_mode == "renew_datastore":
                return self.get_normalized_probs((decoder_out,extra),log_probs=True)
            
            elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["train_metak"]:
                return decoder_out,extra
            
            else:
                return decoder_out



    def get_normalized_probs(self, net_output, log_probs, sample=None, prev_output_tokens=None):

        network_probs = utils.softmax(net_output[0], dim=-1, onnx_trace=self.onnx_trace)
        knn_dists = self.retriever.results["distances"]
        tgt_index = self.retriever.results["vals"]
        knn_key = self.retriever.results["keys"]
        queries = self.retriever.results["query"]
        indices = self.retriever.results["indices"]
        network_select_probs = network_probs.gather(index=tgt_index, dim=-1) # [batch, seq len, K]
        if sample and not prev_output_tokens:
            input = sample['prev_target']
            mask_pos = input.eq(self.unk).sum(dim=1)
            all_pos = input.ne(self.pad).sum(dim=1)
            alpha = mask_pos / all_pos
        elif prev_output_tokens is not None:
            input = prev_output_tokens
            mask_pos = input.eq(self.unk).sum(dim=1)
            all_pos = input.ne(self.pad).sum(dim=1)
            alpha = mask_pos / all_pos



        ###################      train_metak     ####################
        if self.training and self.args.knn_mode == "train_metak":
            knn_dists = torch.sum((knn_key - queries.unsqueeze(-2).detach()) ** 2, dim=-1)   
            knn_dists, new_index = torch.sort(knn_dists, dim=-1)
            tgt_index = tgt_index.gather(dim=-1, index=new_index)
            knn_key = knn_key.gather(dim=-2, index=new_index.unsqueeze(-1).expand(knn_key.shape))

            B, S, K = knn_dists.size()

            network_select_probs = network_probs.gather(index=tgt_index, dim=-1) # [batch, seq len, K]
            target=net_output[1]["target"]
            last_hidden=net_output[1]["last_hidden"]
            random_rate = 0
            noise_var = 0
            e = 1
            import math
            random_rate = random_rate * math.exp((-self.update_num)/e)
            noise_mask = (tgt_index == target.unsqueeze(-1)).any(-1, True)
            rand_mask = ((torch.rand(B, S, 1).cuda() < random_rate) & (target.unsqueeze(-1) != 1)).long()
            rand_mask2 = ((torch.rand(B, S, 1).cuda() < random_rate) & (target.unsqueeze(-1) != 1) & ~noise_mask).float()
                            

            with torch.no_grad():
                # import pdb;pdb.set_trace()
                knn_key = knn_key + torch.randn_like(knn_key) * rand_mask.unsqueeze(-1) * noise_var
                new_key = last_hidden + torch.randn_like(last_hidden) * noise_var
                noise_knn_key = torch.cat([new_key.unsqueeze(-2), knn_key.float()[:, :, :-1, :]], -2)
                noise_tgt_index = torch.cat([target.unsqueeze(-1), tgt_index[:, :, :-1]], -1)               
                tgt_index = noise_tgt_index * rand_mask2.long() + tgt_index * (1 - rand_mask2.long())
                knn_key = noise_knn_key * rand_mask2.unsqueeze(-1) + knn_key * (1 - rand_mask2.unsqueeze(-1))
                
                knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace) # B, S, K, V
                knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)
                noise_knn_dists = torch.sum((knn_key - last_hidden.unsqueeze(-2).detach()) ** 2, dim=3)
                dup_knn_dists = noise_knn_dists 

                # sort the distance again
                new_dists, dist_index = torch.sort(dup_knn_dists, dim=-1)
                new_index = dist_index

                # update the input
                knn_dists = new_dists
                tgt_index = tgt_index.gather(-1, new_index)
                network_select_probs = network_probs.gather(index=tgt_index, dim=-1)
                knn_key_feature = knn_key_feature.gather(-1, new_index)

        ###################      renew     ####################
        elif self.args.knn_mode in ["renew_datastore"]:
            knn_dists = self.retriever.results["distances"]
            tgt_index = self.retriever.results["vals"]
            # knn_key = self.retriever.results["keys"]
            # queries = self.retriever.results["query"]
            indices = self.retriever.results["indices"]
            target = net_output[1]["target"]
            last_hidden=net_output[1]["last_hidden"]
            B, S, K = knn_dists.size()


            valid_mask = target.unsqueeze(-1) > 2
            corr_mask = (tgt_index == target.unsqueeze(-1)) & valid_mask
            # 前K//4中的错误答案
            wrong_mask = torch.full(indices.size(), False).to("cuda")
            wrong_mask[:,:,0:K//4] = (tgt_index[:,:,0:K//4]!= target.unsqueeze(-1)) & valid_mask
            corr_idx = indices[corr_mask]
            wrong_idx = indices[wrong_mask]

            # 前K//4没有正确答案  添加当前key_val
            add_mask = (~(tgt_index[:,:,0:K//4] == target.unsqueeze(-1)).any(-1, True)) & valid_mask
            # 最近值不是正确答案
            # add_mask = (tgt_index[:,:,0]!=target) & valid_mask.squeeze(-1)
            add_mask = add_mask.squeeze(-1)
        
            # 对的太多
            # del_mask = ((tgt_index == target.unsqueeze(-1)).sum(-1) >= K/2)  & valid_mask.squeeze(-1)
            # del_idx = indices[del_mask][:,-1]


            ### 更新 增加检索正确/错误次数  add_mask 增加新key_val   del_mask --
            self.corr_times = self.corr_times.scatter_add_(0, corr_idx, torch.ones_like(corr_idx))
            if hasattr(self.args, "del_wrong") and self.args.del_wrong==True:
                self.wrong_times = self.wrong_times.scatter_add_(0, wrong_idx, torch.ones_like(wrong_idx))
            new_keys = select_keys_with_pad_mask(last_hidden,add_mask)
            new_vals = target[add_mask]
            if len(new_vals)>0:
                self.new_datastore["keys"].add(new_keys.half())
                self.new_datastore["vals"].add(new_vals)

            knn_prob = self.combiner.get_knn_prob(tgt_index,knn_dists, tgt_tokens=target, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob( knn_prob=knn_prob, neural_model_logit=net_output[0],log_probs=True)
            return combined_prob
        
        else:
            # import pdb;pdb.set_trace()
            knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace) # B, S, K, V
            knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)
        knn_prob = self.combiner.get_knn_prob(
                tgt_index=tgt_index,
                knn_dists=knn_dists,
                knn_key_feature=knn_key_feature,
                network_probs=network_probs,
                network_select_probs=network_select_probs,
                alpha=alpha,
                device=net_output[0].device
            )
        combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
        return combined_prob

            
    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            src_embd = encoder_out.encoder_embedding
            src_mask = encoder_out.encoder_padding_mask
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        if self.insertCausalSelfAttn:
            for i, layer in enumerate(self.layers):
                # for layers in insertCausalSelfAttn, every layer needs self attention mask
                self_attn_mask = self.buffered_future_mask(x)
                x, attn, _ = layer(
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=decoder_padding_mask,
                )
                inner_states.append(x)
        else:
            for i, layer in enumerate(self.layers):
                # early exit from the decoder.
                if (early_exit is not None) and (i >= early_exit):
                    break
                x, attn, _ = layer(
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    self_attn_mask=None,
                    self_attn_padding_mask=decoder_padding_mask,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            if self.concatPE:
                x = self.PEfc(torch.cat((x, positions), -1))
            else:
                x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt

    def buffered_one_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        diagonal_mask = torch.zeros([dim, dim])
        # diagonal_mask.fill_(0)
        diagonal_mask.diagonal().fill_(float('-inf'))
        return diagonal_mask



@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer"
)
def base_architecture(args):
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
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
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
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer_wmt_en_de"
)
def nonautoregressive_transformer_wmt_en_de(args):
    base_architecture(args)
