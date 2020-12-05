import torch
import kqat

def get_qconfig(weight_bw, pot):
    rcf_act = kqat.RCF.with_args(
        qscheme=torch.per_tensor_symmetric,
        alpha=10.0,
        alpha_init=kqat.RCFInit.BN_3STD,
        bw=8,
        is_weight=False,
        alpha_pot=pot,
        decay=0.98
    )

    rcf_weight = kqat.RCF.with_args(
        qscheme=torch.per_channel_symmetric,
        alpha=5.0,
        alpha_init=kqat.RCFInit.RT_3STD,
        bw=weight_bw,
        is_weight=True,
        alpha_pot=pot,
        decay=0.9
    )

    rcf_weight_8bit = kqat.RCF.with_args(
        qscheme=torch.per_channel_symmetric,
        alpha=5.0,
        alpha_init=kqat.RCFInit.RT_3STD,
        bw=8,
        is_weight=True,
        alpha_pot=pot,
        decay=0.9
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
        # model.classifier.qconfig = qcfg_8bit
    elif "resnet50" in args.model:
        model.qconfig = qcfg
    elif "efficientnet_lite" in args.model:
        model.qconfig = qcfg
    else:
        raise NotImplementedError("wrong")

    from torch.nn.intrinsic import ConvBn2d, ConvBnReLU2d, ConvReLU2d
    for mod in model.modules():
        if isinstance(mod, (ConvBn2d, ConvBnReLU2d, ConvReLU2d)):
            if mod[0].groups == mod[0].out_channels:
                mod.qconfig = qcfg_8bit
