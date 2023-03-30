"""
PyTorch implementation of the weight matching algorithm from the Git Re-Basin paper by Ainsworth et al. (https://arxiv.org/abs/2209.04836).
Adapted from https://github.com/themrzmaster/git-re-basin-pytorch/blob/5965f4e5697f3b10dc3ed576af36dc08477b2dc6/utils/weight_matching.py
Original code from the paper using JAX: https://github.com/samuela/git-re-basin
Changes compared to themrzmaster/git-re-basin-pytorch by Konstantin Dobler:
- Small bugfixes
- Some explanatory comments that were helpful for me
- Transformers support (moderately tested)
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pprint import pprint
from typing import NamedTuple

import torch
from dargparser import dArg, dargparse
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import AutoModel


@dataclass
class Args:
    mlp: bool = dArg(default=False, help="Test weight matching with MLP.")
    transformer: bool = dArg(
        default=True,
        help="Test weight matching with a BERT/RoBERTa-style transformer. Disable with `--no_transformer` flag.",
    )
    model_name: str = dArg(
        default="roberta-base", help="Model name for HuggingFace transformer."
    )


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def transformer_encoder_permutation_spec(
    add_classifier=False, prefix=""
) -> PermutationSpec:
    """
    Permutation spec for a RoBERTa-style HuggingFace transformer encoder, specifically roberta-base.
    Should be easy to adapt to other sizes. Also works for XLM-RoBERTa and BERT.
    You can use the `prefix` argument to add a prefix to the keys in the permutation spec, e.g. "roberta." if your using a RobertaForMaskedLM model instead of "vanilla" AutoModel.
    """
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }
    embedding = lambda name, p_out: {f"{name}.weight": (None, p_out)}

    def self_attention(name: str, p_in, p_out):
        """
        Query & key out weights are tied in a permutation group but do not effect the value out weights permutation (values are weighted by softmax over dot products between query & key).
        MultiHeadAttention is handled by having a single linear layer that is re-shaped to produce multiple heads. For us this means it works out-of-the box.
        """
        return {
            **dense(f"{name}.query", p_in, f"{p_out}_qk"),
            **dense(f"{name}.key", p_in, f"{p_out}_qk"),
            **dense(f"{name}.value", p_in, p_out),
            # dropout doesn't matter
        }

    def attention_output(name: str, p_in, p_out):
        """
        Here, we have a skip connection from before the attention (prev_hidden + after_hidden before LayerNorm (but after the dense!))
        Therefore, `p_out` needs to be the same permutation group as the input to the self attention layer.
        `p_in` is the permutation group of the output of the self attention layer (which is the **value** output).
        """
        return {
            **dense(f"{name}.dense", p_in, p_out),
            **norm(f"{name}.LayerNorm", p_out),
            # dropout doesn't matter
        }

    def attention(name: str, p_in):
        return {
            **self_attention(f"{name}.self", p_in, f"P_{name}.self.attention_out"),
            **attention_output(f"{name}.output", f"P_{name}.self.attention_out", p_in),
        }

    dense_norm_dropout = lambda name, p_in, p_out: {
        **dense(f"{name}.dense", p_in, p_out),
        **norm(f"{name}.LayerNorm", p_out),
        # dropout doesn't matter
    }

    intermediate = lambda name, p_in, p_out: {
        **dense(f"{name}.dense", p_in, p_out),
        # GELU doesn't matter
    }

    roberta_encoder_layer = lambda name, p: {
        **attention(f"{name}.attention", p),
        **intermediate(
            f"{name}.intermediate", p, f"P_{name}.intermediate.upprojection"
        ),
        **dense_norm_dropout(
            f"{name}.output", f"P_{name}.intermediate.upprojection", p
        ),
    }

    def embeddings(name: str, p_out):
        """
        NOTE: I don't think we want to permute rows (i.e. individual token embeddings) -> that's why `p_in` in embedding is `None`.
        NOTE: Same for position embeddings
        NOTE: Same for token_type_embeddings?
        """
        return {
            **embedding(f"{name}.word_embeddings", p_out),
            **embedding(f"{name}.position_embeddings", p_out),
            **embedding(f"{name}.token_type_embeddings", p_out),
            **norm(f"{name}.LayerNorm", p_out),
        }

    # because of the skip connections in the transformer architecture, a lot of the intermediate layers need to be in the same permutation group
    # i.e. a change in layer 1 also needs to be reflected in layer 2, 3, 4, etc.
    # I called this the "running skip permutation group" (RSPG)
    RSPG = "running_skip_permutation_group"

    axes_to_perm = {**embeddings(f"{prefix}embeddings", RSPG)}

    for i in range(12):
        axes_to_perm.update(roberta_encoder_layer(f"{prefix}encoder.layer.{i}", RSPG))

    if add_classifier:

        def classifier(name: str, p_in):
            return {
                **dense(f"{name}.dense", p_in, f"P_{name}.dense"),
                # dropout doesn't matter
                **dense(f"{name}.out_proj", f"P_{name}.dense", None),
            }

        axes_to_perm.update(classifier("classifier", RSPG))

    return permutation_spec_from_axes_to_perm(axes_to_perm)


###############################################################
############### Reference implementations #####################
################## for MLP, ResNet, VGG #######################
## from https://github.com/themrzmaster/git-re-basin-pytorch ##
###############################################################


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm(
        {
            "layer0.weight": ("P_0", None),
            **{
                f"layer{i}.weight": (f"P_{i}", f"P_{i-1}")
                for i in range(1, num_hidden_layers)
            },
            **{f"layer{i}.bias": (f"P_{i}",) for i in range(num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
            f"layer{num_hidden_layers}.bias": (None,),
        }
    )


"""
def cnn_permutation_spec() -> PermutationSpec:
  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
  dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else  {f"{name}.weight": (p_out, p_in)}

  return permutation_spec_from_axes_to_perm({
     **conv("conv1", None, "P_bg0"),
     **conv("conv2", "P_bg0", "P_bg1"),
     **dense("fc1", "P_bg1", "P_bg2"),
     **dense("fc2", "P_bg2", None, False),
  })
"""


def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock("layer1.2", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock("layer2.2", "P_bg2"),
            # **easyblock("layer2.3", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock("layer3.2", "P_bg3"),
            # **easyblock("layer3.3", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("linear", "P_bg3", None),
        }
    )


# should be easy to generalize it to any depth
def resnet50_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock("layer1.2", "P_bg1"),
            **easyblock("layer1.3", "P_bg1"),
            **easyblock("layer1.4", "P_bg1"),
            **easyblock("layer1.5", "P_bg1"),
            **easyblock("layer1.6", "P_bg1"),
            **easyblock("layer1.7", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock("layer2.2", "P_bg2"),
            **easyblock("layer2.3", "P_bg2"),
            **easyblock("layer2.4", "P_bg2"),
            **easyblock("layer2.5", "P_bg2"),
            **easyblock("layer2.6", "P_bg2"),
            **easyblock("layer2.7", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock("layer3.2", "P_bg3"),
            **easyblock("layer3.3", "P_bg3"),
            **easyblock("layer3.4", "P_bg3"),
            **easyblock("layer3.5", "P_bg3"),
            **easyblock("layer3.6", "P_bg3"),
            **easyblock("layer3.7", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("linear", "P_bg3", None),
        }
    )


def vgg16_permutation_spec() -> PermutationSpec:
    layers_with_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    layers_with_conv_b4 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    layers_with_bn = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
    dense = lambda name, p_in, p_out, bias=True: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }
    return permutation_spec_from_axes_to_perm(
        {
            # first features
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),
            "features.1.running_mean": ("P_Conv_0", None),
            "features.1.running_var": ("P_Conv_0", None),
            "features.1.num_batches_tracked": (),
            **{
                f"features.{layers_with_conv[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    f"P_Conv_{layers_with_conv_b4[i]}",
                    None,
                    None,
                )
                for i in range(len(layers_with_conv))
            },
            **{f"features.{i}.bias": (f"P_Conv_{i}",) for i in layers_with_conv + [0]},
            # bn
            **{
                f"features.{layers_with_bn[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.bias": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.running_mean": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.running_var": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.num_batches_tracked": ()
                for i in range(len(layers_with_bn))
            },
            **dense("classifier", "P_Conv_40", "P_Dense_0", False),
        }
    )


####################################################
############# Weight matching code #################
####################################################


def get_permuted_param(
    ps: PermutationSpec, perm: dict[str, torch.Tensor], k: str, params, except_axis=None
):
    """Get parameter `k` from `params`, with the permutations applied."""
    if not ps.axes_to_perm.get(k):
        print(f"WARNING: No permutation for {k}.")
        return params[k]

    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None, rng=None
):
    """Find a permutation of `params_b` to make them match `params_a`."""
    # pprint(ps.perm_to_axes)

    perm_sizes: dict[str, int] = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    perm = {
        p: torch.arange(n) for p, n in perm_sizes.items()
    }  # if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in tqdm(torch.randperm(len(perm_names), generator=rng)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                if w_a.shape[1] != w_b.shape[1]:
                    # TODO: this is a hack right now if we have word embeddings with different sizes in the two models we want to align, there surely exist more elegant solutions
                    print(
                        f"WARNING: {wk} has different number of features in A and B. Skipping."
                    )
                    continue
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum("ij,ij->i", A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum("ij,ij->i", A, torch.eye(n)[ci, :]).sum()
            print(f"{iteration}/{p}: {newL - oldL}, newL: {newL}, oldL: {oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)

        if not progress:
            break

    return perm


def test_weight_matching_mlp():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=1)
    print(ps.axes_to_perm)
    rng = torch.Generator()
    rng.manual_seed(13)
    num_hidden = 10
    shapes = {
        "layer0.weight": (2, num_hidden),
        "layer0.bias": (num_hidden,),
        "layer1.weight": (num_hidden, 3),
        "layer1.bias": (3,),
    }

    params_a = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    params_b = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    perm = weight_matching(ps, params_a, params_b, rng=rng)
    print(perm)


def test_weight_matching_transformer_encoder(args: Args):
    ps = transformer_encoder_permutation_spec()
    # pprint(ps.axes_to_perm)
    # pprint(ps.perm_to_axes)

    rng = torch.Generator()
    rng.manual_seed(13)
    transformer_model = AutoModel.from_pretrained(args.model_name)
    params_a = transformer_model.state_dict()

    params_b = {
        k: torch.randn(param.shape, generator=rng) for k, param in params_a.items()
    }

    # if we copy params_a, algorithm should terminate immediately and produce the identity permutation
    # params_b = deepcopy(params_a)
    perm = weight_matching(ps, params_a, params_b, rng=rng)
    print(perm)


if __name__ == "__main__":
    args = dargparse(Args)
    transformer_model = AutoModel.from_pretrained(args.model_name)

    # print(transformer_model)
    # print(
    #     transformer_model.encoder.layer[0].attention.output.LayerNorm.weight.shape,
    #     transformer_model.encoder.layer[0].attention.output.LayerNorm.bias.shape,
    # )

    # print(
    #     transformer_model.encoder.layer[0].intermediate.dense.weight.shape,
    #     transformer_model.encoder.layer[0].output.dense.weight.shape,
    # ),

    if args.mlp:
        test_weight_matching_mlp()
    if args.transformer:
        test_weight_matching_transformer_encoder(args)
