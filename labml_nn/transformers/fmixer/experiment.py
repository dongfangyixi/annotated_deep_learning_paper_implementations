"""
---
title: FNet Experiment
summary: This experiment trains a FNet based model on AG News dataset.
---

# [FNet](index.html) Experiment

This is an annotated PyTorch experiment to train a [FNet model](index.html).

This is based on
[general training loop and configurations for AG News classification task](../../experiments/nlp_classification.html).
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_classification import NLPClassificationConfigs
from labml_nn.transformers import Encoder
from labml_nn.transformers import TransformerConfigs
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.optimizers.noam import Noam


class TransformerClassifier(nn.Module):
    """
    # Transformer based classifier model
    """
    def __init__(self, encoder: Encoder, src_embed: Module, generator: nn.Linear):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, None)
        # Get logits for classification.
        #
        # We set the `[CLS]` token at the last position of the sequence.
        # This is extracted by `x[-1]`, where `x` is of
        # shape `[seq_len, batch_size, d_model]`
        x = self.generator(x[-1])

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return x, None


class Configs(NLPClassificationConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPClassificationConfigs`](../../experiments/nlp_classification.html)
    """

    # Classification model
    model: TransformerClassifier
    # Transformer
    transformer: TransformerConfigs


@option(Configs.transformer)
def _transformer_configs(c: Configs):
    """
    ### Transformer configurations
    """

    # We use our
    # [configurable transformer implementation](../configs.html#TransformerConfigs)
    conf = TransformerConfigs()
    # Set the vocabulary sizes for embeddings and generating logits
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens

    #
    return conf


@option(TransformerConfigs.encoder_attn)
def fnet_mix():
    """
    Create `FNetMix` module that can replace the self-attention in
    [transformer encoder layer](../models.html#TransformerLayer)
.
    """
    # from labml_nn.transformers.fmixer import FNetMix
    # from labml_nn.transformers.fmixer import FFT as FNetMix
    # from labml_nn.transformers.fmixer import MIXFFT as FNetMix
    from labml_nn.transformers.fmixer import MIXER as FNetMix
    return FNetMix()


@option(Configs.model)
def _model(c: Configs):
    """
    Create classification model
    """
    m = TransformerClassifier(c.transformer.encoder,
                              c.transformer.src_embed,
                              nn.Linear(c.d_model, c.n_classes)).to(c.device)

    return m



@option(OptimizerConfigs.optimizer, 'Noamwlr')
def _noam_optimizer(c: OptimizerConfigs):
    from labml_nn.optimizers.noam import Noam

    return Noam(c.parameters,
                lr=c.learning_rate, betas=c.betas, eps=c.eps,
                weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad, warmup=c.warmup,
                d_model=c.d_model)

def main():
    # Create experiment
    experiment.create(name="fnet")
    # Create configs
    conf = Configs()

    print("Override configurations")

    experiment.configs(conf, {
        # Use world level tokenizer
        'tokenizer': 'basic_english',

        # Train for $32$ epochs
        'epochs': 32,
        'batch_size': 16,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Transformer configurations (same as defaults)
        'transformer.d_model': 512,
        'transformer.ffn.d_ff': 2048,
        'transformer.n_heads': 8,
        'transformer.n_layers': 6,

        # Use [FNet](index.html) instead of self-a
        # ttention
        'transformer.encoder_attn': 'fnet_mix',

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1e-3,
    })
    print("model: ", conf.model)
    c = OptimizerConfigs()
    para_list = []
    for i in range(6):
        para_list.append({'params': conf.model.encoder.layers[i].feed_forward.parameters()})
        para_list.append({'params': conf.model.encoder.layers[i].self_attn.parameters(), 'lr': 1e-8})
        para_list.append({'params': conf.model.encoder.layers[i].norm_self_attn.parameters()})
        para_list.append({'params': conf.model.encoder.layers[i].norm_ff.parameters()})
    para_list.append({'params': conf.model.src_embed.parameters()})
    para_list.append({'params': conf.model.encoder.norm.parameters()})
    para_list.append({'params': conf.model.generator.parameters()})
    optimizer = Noam(params=para_list,
                     lr=1., betas=c.betas, eps=c.eps,
                     weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad, warmup=c.warmup,
                     d_model=512)

    conf.optimizer = optimizer
    print("optimizer: ", conf.optimizer)

    # exit(0)
    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()
