from transformers import PegasusForConditionalGeneration, PegasusConfig, BartForConditionalGeneration, BartConfig, \
    T5ForConditionalGeneration, T5Config
from transformers.file_utils import ModelOutput

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers.modeling_outputs import BaseModelOutput


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


@dataclass
class Seq2SeqContrastiveOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrast_states: torch.FloatTensor = None


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead
    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PegasusForContrastive(PegasusForConditionalGeneration):
    def __init__(self, config: PegasusConfig, **kwargs):
        super().__init__(config)

        self.classification_head = None
        self.last_token = False
        self.classification_head = ClassificationHead(
            config.d_model,
            config.d_model,
            1024,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        src_select_indices=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
    ):
        # ============
        #  evaluation
        # ============
        if src_select_indices is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ### ENCODER START

        if encoder_outputs is None and src_select_indices is not None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )
            # TODO: what is selected here?
            encoder_outputs = tuple([ele.index_select(0, src_select_indices) for ele in encoder_outputs])
            input_ids = input_ids.index_select(0, src_select_indices)
            attention_mask = attention_mask.index_select(0, src_select_indices)

        ### ENCODER END

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        x = outputs[0]

        if self.classification_head is not None:
            contrastive_x = self.classification_head(x)  # bsz x seqlen x dim
        else:
            contrastive_x = x

        return Seq2SeqContrastiveOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            contrast_states=contrastive_x
        )


class BartForContrastive(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config)

        self.classification_head = None
        self.last_token = False
        self.classification_head = ClassificationHead(
            config.d_model,
            config.d_model,
            1024,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
            self,
            input_ids=None,  # batch_size x seq_len
            pos_neg=None,  # 1 x N binary tensor, 1 means positive target, 0 means negative target
            ref_summ_idx=None,  # 1 x N binary tensor, 1 means reference summary, 0 means augmented targets (positive or negative)
            src_select_indices=None,  # 1 x N, source index of targets
            attention_mask=None,  # batch_size x seq_len
            decoder_input_ids=None,  # N x seq_len
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # ============
        #  evaluation
        # ============
        if src_select_indices is None:
            # print("BART - [Evaluation] forward mode.")
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # some boolean values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ###################
        #     Encoder     #
        ###################
        if encoder_outputs is None:
            # encode the dialogues
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )

            # we need to record the index of dialogue in this batch, for each target summary (either positive or negative)
            # for example, if we have 4 dialogues in a batch (batch_size=4), 16 target summaries (for each dialogue we have
            # 2 positive and 2 negative summaries.
            # dlg_idx: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] means the first 4 target summaries are matching the
            # first dialogue in this batch
            # pos_neg: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] means the first 2 target summaries for the first
            # dialogue are positive summaries (1 for positive, 0 for negative)

            # we replicate encoded dialogues multiple times, to have a 1-1 match between dialogues and target summaries
            # in this case we only encode each dialogue once, to save some computational overhead
            encoder_outputs = tuple([ele.index_select(0, src_select_indices) for ele in encoder_outputs])
            input_ids = input_ids.index_select(0, src_select_indices)
            attention_mask = attention_mask.index_select(0, src_select_indices)

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        ###################
        #     Decoder     #
        ###################
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # shared.weight is from Embedding layer
        x = outputs[0]  # N x seq_len x hidden_dim (e.g., 1024 for bart large)
        # N x seq_len x vocab_size
        lm_logits = F.linear(x, self.model.shared.weight, bias=self.final_logits_bias)

        if self.classification_head is not None:
            contrastive_x = self.classification_head(x)  # bsz x seqlen x dim
        else:
            contrastive_x = x

        return Seq2SeqContrastiveOutput(
            loss=None,  # a placeholder to make sure prediction step doesn't break
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            contrast_states=contrastive_x
        )


class T5ForContrastive(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)

        config.classifier_dropout = 0

        self.classification_head = None
        self.last_token = False
        self.classification_head = ClassificationHead(
            config.d_model,
            config.d_model,
            1024,
            config.classifier_dropout,
        )
        self.encoder._init_weights(self.classification_head.dense)
        self.encoder._init_weights(self.classification_head.out_proj)

    def forward(
            self,
            input_ids=None,  # batch_size x seq_len
            pos_neg=None,  # 1 x N binary tensor, 1 means positive target, 0 means negative target
            ref_summ_idx=None,  # 1 x N binary tensor, 1 means reference summary, 0 means augmented targets (positive or negative)
            src_select_indices=None,  # 1 x N, source index of targets
            attention_mask=None,  # batch_size x seq_len
            decoder_input_ids=None,  # N x seq_len
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # some boolean values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ###################
        #     Encoder     #
        ###################
        if encoder_outputs is None:
            # encode the dialogues
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )

            # we need to record the index of dialogue in this batch, for each target summary (either positive or negative)
            # for example, if we have 4 dialogues in a batch (batch_size=4), 16 target summaries (for each dialogue we have
            # 2 positive and 2 negative summaries.
            # dlg_idx: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] means the first 4 target summaries are matching the
            # first dialogue in this batch
            # pos_neg: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] means the first 2 target summaries for the first
            # dialogue are positive summaries (1 for positive, 0 for negative)

            # we replicate encoded dialogues multiple times, to have a 1-1 match between dialogues and target summaries
            # in this case we only encode each dialogue once, to save some computational overhead
            encoder_outputs = tuple([ele.index_select(0, src_select_indices) for ele in encoder_outputs])
            input_ids = input_ids.index_select(0, src_select_indices)
            attention_mask = attention_mask.index_select(0, src_select_indices)

        # we wrap the encoder output in a BaseModelOutput
        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        ###################
        #     Decoder     #
        ###################
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # shared.weight is from Embedding layer
        decoder_last_hidden_state = decoder_outputs[0]  # N x seq_len x hidden_dim (e.g., 1024 for bart large)
        # N x seq_len x vocab_size
        # lm_logits = F.linear(decoder_last_hidden_state, self.model.shared.weight, bias=self.final_logits_bias)
        lm_logits = self.lm_head(decoder_last_hidden_state)

        x = decoder_outputs[0]
        if self.classification_head is not None:
            contrastive_x = self.classification_head(x)  # bsz x seqlen x dim
        else:
            contrastive_x = x

        return Seq2SeqContrastiveOutput(
            loss=torch.FloatTensor([-1]),  # a placeholder to make sure prediction step doesn't break
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            contrast_states=contrastive_x
        )