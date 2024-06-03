
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel,NATransformerDecoder
from fairseq.utils import new_arange
from fairseq import utils
import math
import torch.nn.functional as F
import torch
from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    disable_model_grad,
    enable_module_grad,
    archs,
    select_correct_position_mask,select_wrong_position_mask,select_all_position_mask
)
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import RobustCombiner


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def _random_mask(target_tokens,p):
    target_masks = (
        target_tokens.ne(1) & target_tokens.ne(0) & target_tokens.ne(2)
    )
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    target_length = target_length * target_length.clone().uniform_()
    target_length = target_length + 1  # make sure to mask at least one token.

    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    prev_target_tokens = target_tokens.masked_fill(
        target_cutoff.scatter(1, target_rank, target_cutoff), 3
    )
    return prev_target_tokens

@register_model("robust_cmlmc")
class ROBUSTCMLMC(NATransformerModel):
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
        parser.add_argument("--knn-max-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
                            help="The directory to save/load robustCombiner")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
        parser.add_argument("--mask-mode",type=str,)
        parser.add_argument("--knn-inference-mode",type=str,default="all")
        ######################################## KNN arguments ####################################################
        # ? hyper-params for robust training 
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
        #############################################################################################################
        parser.add_argument("--new-datastore", type=str, metavar="STR", default="_new",
                            help="The directory to save/load robustCombiner")
    


    def __init__(self, args, encoder, decoder):
        # import pdb;pdb.set_trace()
        super().__init__(args, encoder, decoder)
        ######################################## CMLMC Modifications ####################################################
        self.selfcorrection = args.selfcorrection
        self.correctingself = False
        self.replacefactor = args.replacefactor
        ######################################## CMLMC Modifications ####################################################
        if hasattr(self.args, "knn_mode") and args.knn_mode == "train_metak":
            # import pdb;pdb.set_trace()
            disable_model_grad(self)
            enable_module_grad(self, "combiner")

            

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with RobustKNNMTDecoder
        """
        return RobustCMLMCDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, update_num=None, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."
        # import pdb;pdb.set_trace()
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # import pdb;pdb.set_trace()
        decoder_out = self.decoder(normalize=True, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, tgt_tokens=tgt_tokens)
        word_ins_mask = prev_output_tokens.eq(self.unk)

        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "train_metak":
            # decoder_out = self.forward_decoder(decoder_out=decoder_out, encoder_out=encoder_out, tgt_tokens=tgt_tokens)
            return decoder_out
        
        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "renew_datastore":
            # import pdb;pdb.set_trace()
            word_ins_out = decoder_out
            output_tokens = decoder_out.max(-1)[1]
            word_ins_mask = tgt_tokens.ne(self.pad) & tgt_tokens.ne(self.eos) & tgt_tokens.ne(self.bos)
            remask_mask = (output_tokens!=tgt_tokens) & word_ins_mask 
            if remask_mask.sum()!=0:
                prev_output_tokens = tgt_tokens.masked_fill(remask_mask, self.unk)
                decoder_out = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out,tgt_tokens=tgt_tokens)
                word_ins_out = decoder_out
                word_ins_mask = remask_mask



        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "build_datastore" and "iteration" in self.args.mask_mode:
            step = 1
            max_step = min(10,word_ins_mask.sum(-1).clamp_(min=1).min())
            # max_step = len(output_tokens[0])-1
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
                    step=step
                ).max(-1)
                step+=1
        
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


class RobustCMLMCDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.update_num = 0

        if hasattr(self.args, "knn_mode") and args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["inference","train_metak","renew_datastore"]:
            # import pdb;pdb.set_trace()
            if "datastore" not in global_vars():
                global_vars()['datastore'] = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
            self.datastore = global_vars()['datastore'] 
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            if args.knn_mode == "train_metak":
                self.combiner = RobustCombiner(
                    max_k=args.knn_max_k, 
                    midsize=args.robust_wp_hidden_size, 
                    midsize_dc=args.robust_dc_hidden_size, 
                    topk_wp=args.robust_wp_topk, 
                    probability_dim=len(dictionary)
                )

            if args.knn_mode == "renew_datastore":
                self.combiner = RobustCombiner.load(args.knn_combiner_path)
                if "corr_times" not in global_vars():
                    global_vars()["corr_times"] = torch.zeros_like(torch.tensor(self.datastore["vals"].data)).to("cuda")
                    global_vars()["wrong_times"] = torch.zeros_like(torch.tensor(self.datastore["vals"].data)).to("cuda")
                self.corr_times =   global_vars()["corr_times"]
                self.wrong_times =  global_vars()["wrong_times"]
                if "new_datastore" not in global_vars():
                    global_vars()["new_datastore"] = Datastore(args.knn_datastore_path+args.new_datastore)  
                self.new_datastore = global_vars()["new_datastore"]


           
            elif args.knn_mode == "inference":
                self.combiner = RobustCombiner.load(args.knn_combiner_path)
    

    def forward(self, normalize, encoder_out, prev_output_tokens, tgt_tokens=None, step=0, **unused):

        x, extra= self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        # import pdb;pdb.set_trace()
        if hasattr(self.args, "knn_mode") and self.args.knn_mode == "build_datastore":
            _score,_tokens = self.output_layer(x).max(-1)
            unk_mask = prev_output_tokens.eq(self.unk)
            token_mask = tgt_tokens.ne(self.pad)  & tgt_tokens.ne(self.bos) & tgt_tokens.ne(self.eos)
            mask=None
            keys=None
            if "withoutfirst" in self.args.mask_mode and step == 0:
                pass
            elif "corr" in self.args.mask_mode:
                mask, keys = select_correct_position_mask(x,_tokens,tgt_tokens,token_mask)
            elif "wrong" in self.args.mask_mode:
                mask, keys = select_wrong_position_mask(x,_tokens,tgt_tokens,token_mask)
            elif "all" in self.args.mask_mode:
                ### tgt --> prev 11.8   ### add tgt.ne(self.pad) 11.13 
                mask = prev_output_tokens.ne(self.pad) & prev_output_tokens.ne(self.bos) \
                    & prev_output_tokens.ne(self.eos) & tgt_tokens.ne(self.pad) \
                    & tgt_tokens.ne(self.bos) & tgt_tokens.ne(self.eos)    
                keys = select_all_position_mask(x,_tokens,mask)
            else:
                mask = unk_mask
                keys = select_keys_with_pad_mask(x,unk_mask)
    
            vals = tgt_tokens[mask]
            if (keys==None) or (mask==None):
                pass
            else:
                self.datastore["keys"].add(keys.half())
                self.datastore["vals"].add(vals)
                steps = torch.ones_like(vals)
                self.datastore["steps"].add(steps.fill_(step))

        elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["inference","train_metak","renew_datastore"]:
            self.retriever.retrieve(x, return_list=["vals", "query", "distances", "keys","indices"]) 
        extra.update({"last_hidden": x, "target": tgt_tokens})

        decoder_out = self.output_layer(x)

        if normalize:
            if hasattr(self.args, "knn_mode") and self.args.knn_mode in ["inference","renew_datastore"]:
                # import pdb;pdb.set_trace()
                return self.get_normalized_probs((decoder_out,extra),log_probs=True)
            elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["train_metak"]:
                return decoder_out,extra
            else:
                return F.log_softmax(decoder_out, -1)
        else:
            if "con_cmlm" in self.args.arch or "cl" in self.args.arch:
                return decoder_out,x.clone()
            elif hasattr(self.args, "knn_mode") and self.args.knn_mode in ["renew_datastore"]:
                # import pdb;pdb.set_trace()
                mask = prev_output_tokens.eq(self.unk)
                if mask.sum()==0:
                    return decoder_out
                keys = select_keys_with_pad_mask(x,mask)
                vals = tgt_tokens[mask]
                self.new_datastore["keys"].add(keys.half())
                self.new_datastore["vals"].add(vals)
                return decoder_out
            else:
                return decoder_out



    def get_normalized_probs(
            self,
            net_output,
            log_probs,
            sample= None,
        ):
            r"""
            we overwrite this function to change the probability calculation process.
            step 1. 
                calculate the knn probability based on retrieved resultes
            step 2.
                combine the knn probability with NMT's probability 
            """
            if self.args.knn_mode in ["inference","train_metak","renew_datastore"] :
                # import pdb;pdb.set_trace()
                network_probs = utils.softmax(net_output[0], dim=-1, onnx_trace=self.onnx_trace)
                knn_dists = self.retriever.results["distances"]
                tgt_index = self.retriever.results["vals"]
                knn_key = self.retriever.results["keys"]
                queries = self.retriever.results["query"]
                indices = self.retriever.results["indices"]

                knn_dists = torch.sum((knn_key - queries.unsqueeze(-2).detach()) ** 2, dim=-1)   
                knn_dists, new_index = torch.sort(knn_dists, dim=-1)
                tgt_index = tgt_index.gather(dim=-1, index=new_index)
                knn_key = knn_key.gather(dim=-2, index=new_index.unsqueeze(-1).expand(knn_key.shape))

                B, S, K = knn_dists.size()
                # 选出knn结果的对应模型概率值
                network_select_probs = network_probs.gather(index=tgt_index, dim=-1) # [batch, seq len, K]


                ###################      train_metak     ####################
                if self.training and self.args.knn_mode == "train_metak":
                    target=net_output[1]["target"]
                    last_hidden=net_output[1]["last_hidden"]
                    random_rate = self.args.robust_training_alpha0
                    noise_var = self.args.robust_training_sigma
                    e = self.args.robust_training_beta
                    random_rate = random_rate * math.exp((-self.update_num)/e)
                    noise_mask = (tgt_index == target.unsqueeze(-1)).any(-1, True)
                    rand_mask = ((torch.rand(B, S, 1).cuda() < random_rate) & (target.unsqueeze(-1) != 1)).long()
                    rand_mask2 = ((torch.rand(B, S, 1).cuda() < random_rate) & (target.unsqueeze(-1) != 1) & ~noise_mask).float()
                                    
                    with torch.no_grad():
                        # add perturbation
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
                    target = net_output[1]["target"]
                    last_hidden=net_output[1]["last_hidden"]
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


                    # 全对 删除最远
                    # del_mask = ((tgt_index == target.unsqueeze(-1)).sum(-1) == K)  & valid_mask.squeeze(-1)
                    # del_idx = indices[del_mask][:,-1]


                    ### 更新 增加检索正确/错误次数  add_mask 增加新key_val   del_mask --
                    self.corr_times = self.corr_times.scatter_add_(0, corr_idx, torch.ones_like(corr_idx))
                    # if hasattr(self.args, "new_datastore") and "del_wrong_answer" in self.args.new_datastore:
                    self.wrong_times = self.wrong_times.scatter_add_(0, wrong_idx, torch.ones_like(wrong_idx))
                    new_keys = select_keys_with_pad_mask(last_hidden,add_mask)
                    new_vals = target[add_mask]
                    if len(new_vals)>0:
                        # import pdb;pdb.set_trace()
                        self.new_datastore["keys"].add(new_keys.half())
                        self.new_datastore["vals"].add(new_vals)
                    # self.corr_times = torch.cat(self.corr_times,torch.ones_like(new_vals))
                    # self.wrong_times = torch.cat(self.wrong_times,torch.zeros_like(new_vals))
                    # self.corr_times = self.corr_times.scatter_add_(0, del_idx, torch.full(del_idx.size(),-1).to("cuda"))

                    # import pdb;pdb.set_trace()
                    with torch.no_grad():
                        # add perturbation
                        added_keys = last_hidden
                        cat_keys = torch.cat([added_keys.unsqueeze(-2), knn_key.float()[:, :, :-1, :]], -2)
                        cat_targets = torch.cat([target.unsqueeze(-1), tgt_index[:, :, :-1]], -1)  
                        tgt_index = add_mask.unsqueeze(-1) * cat_targets + ~(add_mask.unsqueeze(-1)) * tgt_index
                        knn_key = add_mask.unsqueeze(-1).unsqueeze(-1) * cat_keys + ~(add_mask.unsqueeze(-1).unsqueeze(-1)) * knn_key


                        knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace) # B, S, K, V
                        knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)

                        ### 重排
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
                    
                        
                else:
                    knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace) # B, S, K, V
                    knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)

                knn_prob = self.combiner.get_knn_prob(
                    tgt_index=tgt_index,
                    knn_dists=knn_dists,
                    knn_key_feature=knn_key_feature,
                    network_probs=network_probs,
                    network_select_probs=network_select_probs,
                    device=net_output[0].device
                )
                combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
                return combined_prob
            else:
                return F.log_softmax(net_output, -1)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        # import pdb;pdb.set_trace()
        self.update_num = num_updates

@register_model_architecture("robust_cmlmc", "robust_cmlmc")
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


@register_model_architecture("robust_cmlmc", "robust_cmlmc_wmt_en_de")
def cmlmc_wmt_en_de(args):
    cmlmc_base_architecture(args)

@register_model_architecture("robust_cmlmc", "robust_cmlmc@transformer")
def cmlmc_knn_wmt_en_de(args):
    cmlmc_base_architecture(args)


from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion("label_smoothed_cross_entropy_for_robust_cmlmc")
class LabelSmoothedCrossEntropyCriterionForRobustCMLMC(
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
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # import pdb;pdb.set_trace()
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
        lprobs = model.get_normalized_probs(net_output, log_probs=True,sample=sample)
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