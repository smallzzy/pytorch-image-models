import torch
import kqat

def get_qconfig(weight_bw, pot, bitwidth_range=[8.,8.,8.], symmetric_clipping=False):
    qmin, qmax = kqat.get_min_max(8)

    rcf_act = kqat.KAQ.with_args(
        qscheme=torch.per_tensor_symmetric,
        # qmin=qmin, qmax=qmax,
        bitwidth_range=bitwidth_range,
        symmetric_clipping=symmetric_clipping,
        init=kqat.RadixInit.BN_3STD|kqat.RadixInit.RT_MINMAX,
        threshold=6,
        is_weight=False,
        pot=pot,
        decay=0.98
    )

    # rcf_weight_8bit = kqat.KAQ.with_args(
    #     qscheme=torch.per_channel_symmetric,
    #     qmin=qmin, qmax=qmax,
    #     bitwidth_range=bitwidth_range,
    #     symmetric_clipping=symmetric_clipping,
    #     init=kqat.RadixInit.RT_MINMAX,
    #     threshold=6,
    #     is_weight=True,
    #     pot=pot,
    #     decay=0.98
    # )

    # qmin, qmax = kqat.get_min_max(weight_bw)

    rcf_weight = kqat.KAQ.with_args(
        qscheme=torch.per_channel_symmetric,
        # qmin=qmin, qmax=qmax,
        bitwidth_range=bitwidth_range,
        symmetric_clipping=symmetric_clipping,
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

    # qcfg_8bit = kqat.KQConfig(
    #     activation=rcf_act,
    #     weight=rcf_weight_8bit,
    #     bias=torch.nn.Identity)

    return qcfg

def attach_qconfig(args, model):
    qcfg = get_qconfig(args.bitwidth, args.pot, args.bitwidth_range, args.symmetric_clipping)
    # qconfig attaching is always model dependent?
    if "mobilenetv2" in args.model:
        model.qconfig = qcfg
        model.conv_stem.qconfig = qcfg
        model.classifier.qconfig = qcfg
    elif "resnet50" in args.model:
        model.qconfig = qcfg
    else:
        raise NotImplementedError("wrong")
