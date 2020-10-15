import torch
import kqat

def get_qconfig(weight_bw, pot):
    rcf_act = kqat.RCF.with_args(
        alpha=10.0,
        alpha_init=kqat.RCFInit.BN_3STD,
        bw=8,
        is_weight=True,
        alpha_pot=pot
    )

    rcf_weight = kqat.RCF.with_args(
        alpha=5.0,
        alpha_init=kqat.RCFInit.RT_3STD,
        bw=weight_bw,
        is_weight=True,
        alpha_pot=pot
    )


    rcf_weight_8bit = kqat.RCF.with_args(
        alpha=5.0,
        alpha_init=kqat.RCFInit.RT_3STD,
        bw=8,
        is_weight=True,
        alpha_pot=pot
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
    else:
        raise NotImplementedError("wrong")
