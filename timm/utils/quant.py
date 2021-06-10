import torch
import kqat

def get_qconfig(weight_bw, pot, bitwidth_range=None, symmetric_clipping=False, gradient_bias_option='PartialNoise', gradient_start_possibility=0.3):
    if bitwidth_range is None:
        bitwidth_range = [8.]
    qmin, qmax = kqat.get_min_max(8)

    rcf_act = kqat.KAQ.with_args(
        qscheme=torch.per_tensor_symmetric,
        bitwidth_init=weight_bw,
        bitwidth_range=bitwidth_range,
        symmetric_clipping=symmetric_clipping,
        gradient_bias_option=gradient_bias_option, 
        gradient_start_possibility=gradient_start_possibility,
        init=kqat.RadixInit.BN_3STD|kqat.RadixInit.RT_MINMAX,
        threshold=6,
        is_weight=False,
        pot=pot,
        decay=0.98
    )

    rcf_weight = kqat.KAQ.with_args(
        qscheme=torch.per_channel_symmetric,
        bitwidth_init=weight_bw,
        bitwidth_range=bitwidth_range,
        symmetric_clipping=symmetric_clipping,
        gradient_bias_option=gradient_bias_option, 
        gradient_start_possibility=gradient_start_possibility,
        init=kqat.RadixInit.RT_MINMAX,
        threshold=6,
        is_weight=True,
        pot=pot,
        decay=0.98
    )

    qcfg = kqat.KQConfig(
        activation=rcf_act,
        weight=rcf_weight,
        bias=torch.nn.Identity)
    return qcfg

def attach_qconfig(args, model):
    qcfg = get_qconfig(args.bitwidth, args.pot, args.bitwidth_range, args.symmetric_clipping, args.gradient_bias_option, args.gradient_start_possibility)
    # qconfig attaching is always model dependent?
    if "mobilenetv2" in args.model:
        model.qconfig = qcfg
        model.conv_stem.qconfig = qcfg
        model.classifier.qconfig = qcfg
    elif "resnet50" in args.model:
        model.qconfig = qcfg
    else:
        raise NotImplementedError("wrong")
