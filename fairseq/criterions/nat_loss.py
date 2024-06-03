# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor


@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        # import pdb;pdb.set_trace()
        if type(outputs) is list:
            outputs_list = outputs
            masks_list = masks
            loss = 0
            nll_loss = 0
            i = 0.0

            for (outputs, masks) in zip(outputs_list, masks_list):
                if masks is not None:
                    if torch.sum(torch.isnan(outputs)).item():
                        assert torch.sum(torch.isnan(outputs)).item() == 0
                    outputs, temp_targets = outputs[masks], targets[masks]
                if masks is not None and not masks.any():
                    temp_nll_loss = torch.tensor(0)
                    temp_loss = temp_nll_loss
                else:
                    i += 1.0
                    logits = F.log_softmax(outputs, dim=-1)

                    if temp_targets.dim() == 1:
                        losses = F.nll_loss(logits, temp_targets.to(logits.device), reduction='none')
                    else:  # soft-labels
                        losses = F.kl_div(logits, temp_targets.to(logits.device), reduction='none')
                        losses = losses.sum(-1)
                    temp_nll_loss = mean_ds(losses)

                    if torch.sum(torch.isnan(temp_nll_loss)).item():
                        assert torch.sum(torch.isnan(temp_nll_loss)).item() == 0
                    if label_smoothing > 0:
                        temp_loss = temp_nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                    else:
                        temp_loss = temp_nll_loss
                loss = loss + temp_loss
                nll_loss = nll_loss + temp_nll_loss
            loss = loss / i
            nll_loss = nll_loss / i
        else:
            if masks is not None:
                outputs, targets = outputs[masks], targets[masks]

            if masks is not None and not masks.any():
                nll_loss = torch.tensor(0)
                loss = nll_loss
            else:
                logits = F.log_softmax(outputs, dim=-1)
                if targets.dim() == 1:
                    losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
                else:  # soft-labels
                    losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
                    losses = losses.sum(-1)

                nll_loss = mean_ds(losses)
                if label_smoothing > 0:
                    loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                else:
                    loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}
    
    def _label_smoothed_nll_loss(self,contrastive_scores, contrastive_labels,name=None, eps=0.0): # contrastive_labels: bsz x seqlen; masked position with 0., otherwise 1.
        '''
            contrasive_scores: bsz x seqlen x seqlen
            contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.
        '''
        # import pdb;pdb.set_trace()
        # bsz, seqlen, _ = contrastive_scores.size()
        # logprobs = -F.log_softmax(contrastive_scores, dim=-1)
        # logprobs[~contrastive_scores.bool()] = 0 
        # diagonal_elements = torch.diagonal(logprobs, offset=0, dim1=-2, dim2=-1)
        # loss = torch.sum(diagonal_elements) / contrastive_labels.sum()
        bsz, seqlen, _ = contrastive_scores.size()
        logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
        gold = torch.arange(seqlen).view(-1,)
        gold = gold.expand(bsz, seqlen).contiguous().view(-1)
        if contrastive_scores.is_cuda:
            gold = gold.cuda(contrastive_scores.get_device())
        loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
        loss = loss.view(bsz, seqlen) * contrastive_labels
        loss = torch.sum(loss) / contrastive_labels.sum()

        # _, pred = torch.max(logprobs, -1)
        # correct_num = torch.eq(gold, pred).float().view(bsz, seqlen)
        # correct_num = torch.sum(correct_num * contrastive_labels)

        return {"name": name, "loss": loss,  "factor": 1}


    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True, update_num=None):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        # import pdb;pdb.set_trace()
        if update_num is None:
            outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        else:
            outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, update_num=update_num)
        losses, nll_loss = [], []

        for obj in outputs:
            if "con_loss" in obj:
                if outputs[obj].get("contrastive_scores") == None:
                    continue
                else:
                    _losses = self._label_smoothed_nll_loss(
                        outputs[obj].get("contrastive_scores"),
                        outputs[obj].get("contrastive_labels"),
                        name=obj +"-loss",
                    )
            elif outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
