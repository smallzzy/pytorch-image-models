import torch
import kqat

def get_qconfig(weight_bw, pot):
    qmin, qmax = kqat.get_min_max(8)

    rcf_act = kqat.KAQ.with_args(
        qscheme=torch.per_tensor_symmetric,
        qmin=qmin, qmax=qmax,
        init=kqat.RadixInit.BN_3STD|kqat.RadixInit.RT_MINMAX,
        threshold=6,
        is_weight=False,
        pot=pot,
        decay=0.98
    )

    rcf_weight_8bit = kqat.KAQ.with_args(
        qscheme=torch.per_channel_symmetric,
        qmin=qmin, qmax=qmax,
        init=kqat.RadixInit.RT_MINMAX,
        threshold=6,
        is_weight=True,
        pot=pot,
        decay=0.98
    )

    qmin, qmax = kqat.get_min_max(weight_bw)

    rcf_weight = kqat.KAQ.with_args(
        qscheme=torch.per_channel_symmetric,
        qmin=qmin, qmax=qmax,
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

    qcfg_8bit = kqat.KQConfig(
        activation=rcf_act,
        weight=rcf_weight_8bit,
        bias=torch.nn.Identity)

    return qcfg, qcfg_8bit

def attach_qconfig(args, model):
    qcfg, qcfg_8bit = get_qconfig(args.bitwidth, args.pot)
    # qconfig attaching is always model dependent?
    if "mobilenetv2" in args.model:
        model.qconfig = qcfg
        model.conv_stem.qconfig = qcfg_8bit
        model.classifier.qconfig = qcfg_8bit
    elif "resnet50" in args.model:
        model.qconfig = qcfg
    else:
        raise NotImplementedError("wrong")
