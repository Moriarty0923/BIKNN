r""" 
This file is copied from fairseq_cli/validate.py.
knnbox made 2 major changes:

change 1. We modified the part of parsing args so that is
can parse the arch specified on the cli instead of directly
using the arch inside checkpoint.

change 2. we add codes about `saving datastore vals`, `dump datastore`, etc. 
"""

import logging
import os
import sys
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar

## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from knnbox.datastore import Datastore, GreedyMergeDatastore
from knnbox.common_utils import filter_pad_tokens, global_vars
import numpy as np
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]
    
    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.knn_mode == "build_datastore":
        knn_type = args.arch.split("@")[0]
        if "datastore" not in global_vars():
            # create suitable datastore class if not exists
            if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt"]:
                global_vars()["datastore"] = Datastore(path=args.knn_datastore_path)
           
        datastore = global_vars()["datastore"]
    
    elif args.knn_mode == "renew_datastore":
        if "datastore" not in global_vars():
            global_vars()['datastore'] = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
        datastore = global_vars()['datastore']
        if "corr_times" not in global_vars():
            global_vars()["corr_times"] = torch.zeros_like(torch.tensor(datastore["vals"].data)).to("cuda")
            global_vars()["wrong_times"] = torch.zeros_like(torch.tensor(datastore["vals"].data)).to("cuda")

        if "new_datastore" not in global_vars():
            global_vars()["new_datastore"] = Datastore(args.knn_datastore_path+args.new_datastore)  

        # import pdb;pdb.set_trace()
        # old_datastore = global_vars()['datastore']
        # good_mask = global_vars()["corr_times"]>=global_vars()["wrong_times"]
        # good_keys = torch.tensor(old_datastore['keys'].data).cuda()[good_mask]
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end



    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if args.task == "translation_lev":
                # import pdb;pdb.set_trace()
                _loss, _sample_size, log_output = task.valid_step(sample, model, criterion, datastore)
                
            else:
                if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt"]:
                    non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                    datastore["vals"].add(non_pad_tokens)
                    datastore.set_pad_mask(mask)
    
                ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
                _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)

            progress.log(log_output, step=i)
            log_outputs.append(log_output)

        # import pdb;pdb.set_trace()
        if args.distributed_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=getattr(args, "all_gather_list_size", 16384),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)
    

    # knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # release memory to make sure we have enough gpu memory to build faiss index
    del model, task, progress, criterion, dataset
    if use_cuda:
        torch.cuda.empty_cache()    # release gpu memory
    if args.knn_mode == "build_datastore":
        if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "robust_knn_mt"]:
            # import pdb;pdb.set_trace()
            datastore.dump()
            datastore.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))   # build faiss index
              
    elif args.knn_mode == "renew_datastore":
        new_datastore = global_vars()["new_datastore"]
        corr_times = global_vars()['corr_times']
        wrong_times = global_vars()['wrong_times']
        old_datastore = global_vars()['datastore']
        good_mask = (corr_times>=wrong_times).cpu()
        good_keys = torch.tensor(old_datastore['keys'].data)[good_mask]
        good_vals = torch.tensor(old_datastore['vals'].data)[good_mask]
        logger.info("bad_vals_numbers: {} ".format((~good_mask).sum()))
        logger.info("new_vals_numbers: {} ".format(new_datastore['vals'].data.nonzero()[-1][-1]))

        new_datastore['keys'].add(good_keys)
        new_datastore['vals'].add(good_vals)
        new_datastore.dump()
        new_datastore.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))   # build faiss index
       
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


## knnbox code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_build_datastore_parser(default_task=None):
    r"""
    very similar to options.get_validation_parser() but parse arch as well.

    Difference:
    - when validate, we don't need to specify --arch and model args, because they are
    recorded in .pt file.

    - when building datastore, we need to load the saved model parameter to a knn-mt arch,
    which is different from the checkpoint original arch.
    For example, I have a nmt checkpoint with arch `transformer_iwslt_de_en`, and now I want to
    load it's parameter to arch `vanilla@transformer_iwslt_de_en`, I must specify
    arch = "vanilla@transfromer_iwslt_de_en".
    """
    parser = options.get_parser("Validation", default_task)
    options.add_dataset_args(parser, train=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    # knnbox add one line below to parse arch
    options.add_model_args(parser)
    group = parser.add_argument_group("Evaluation")
    from fairseq.dataclass.data_class import CommonEvalParams
    options.gen_parser_from_dataclass(group, CommonEvalParams())
    return parser
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end 

def cli_main():
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # parser = options.get_validation_parser()
    parser = get_build_datastore_parser()
    args = options.parse_args_and_arch(parser)


    ## only override args that are explicitly given on the command line
    # override_parser = options.get_validation_parser()
    override_parser = get_build_datastore_parser()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)




if __name__ == "__main__":
    cli_main()
    

