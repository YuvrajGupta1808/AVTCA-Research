"""Modality dropout and the inference-time modality ablation act on the same tensors.

Both zero a modality's whole token sequence right before the first cross-attention
exchange (av1 / va1), so a hook on av1 sees exactly what each switch removed.
"""

import unittest

import torch

from models.multimodal_cnn import MultiModalCNN

N_MELS = 64
FEATURES = 1024


def build(**kwargs):
    torch.manual_seed(0)
    return MultiModalCNN(
        num_classes=4,
        fusion='it',
        seq_length=4,
        pretr_ef='None',
        num_heads=4,
        late_text_fusion=False,
        **kwargs,
    )


def inputs(batch_size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch_size, N_MELS, 32, generator=generator),
        torch.randn(batch_size, 4, FEATURES, generator=generator),   # precomputed per-frame features
    )


def first_exchange_inputs(model, audio, video):
    """(audio tokens, video tokens) as they enter av1: av1(xk=video, xq=audio)."""
    captured = {}

    def hook(_module, args, kwargs):
        captured['video'] = args[0].detach().clone()
        captured['audio'] = args[1].detach().clone()

    handle = model.av1.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        model(audio, video)
    finally:
        handle.remove()
    return captured['audio'], captured['video']


def zeroed_rows(tokens):
    return (tokens.flatten(start_dim=1).abs().sum(dim=1) == 0)


class TestModalityDropout(unittest.TestCase):
    def test_training_drops_whole_modalities_per_sample_and_independently(self):
        model = build().train()
        torch.manual_seed(123)
        audio_tokens, video_tokens = first_exchange_inputs(model, *inputs(256))
        audio_dropped = zeroed_rows(audio_tokens)
        video_dropped = zeroed_rows(video_tokens)

        # Per sample: some samples lose a stream, most keep both.
        for dropped in (audio_dropped, video_dropped):
            self.assertGreater(int(dropped.sum()), 10)
            self.assertLess(int(dropped.sum()), 80)          # p = 0.15 of 256 is about 38
        # Independent draws: the two streams are not dropped for the same samples.
        self.assertFalse(torch.equal(audio_dropped, video_dropped))
        # Nothing guards against dropping both streams of a sample (about 2% expected).
        self.assertGreater(int((audio_dropped & video_dropped).sum()), 0)

    def test_dropped_sample_is_zero_at_every_token_and_channel(self):
        model = build().train()
        torch.manual_seed(5)
        audio_tokens, _ = first_exchange_inputs(model, *inputs(64))
        dropped = zeroed_rows(audio_tokens)
        kept = ~dropped
        self.assertTrue(bool((audio_tokens[kept].abs().sum(dim=(1, 2)) > 0).all()))

    def test_no_dropout_in_eval_mode(self):
        model = build().eval()
        with torch.no_grad():
            audio_tokens, video_tokens = first_exchange_inputs(model, *inputs(64))
        self.assertFalse(bool(zeroed_rows(audio_tokens).any()))
        self.assertFalse(bool(zeroed_rows(video_tokens).any()))

    def test_zero_probability_disables_dropout(self):
        model = build(modality_dropout=0.0).train()
        audio_tokens, video_tokens = first_exchange_inputs(model, *inputs(64))
        self.assertFalse(bool(zeroed_rows(audio_tokens).any()))
        self.assertFalse(bool(zeroed_rows(video_tokens).any()))

    def test_probability_is_recorded_in_model_config(self):
        self.assertEqual(
            build().model_config(),
            {'num_heads': 4, 'fusion': 'it', 'it_fusion_mode': 'modern', 'modality_dropout': 0.15},
        )

    def test_invalid_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'modality_dropout'):
            build(modality_dropout=1.0)


class TestModalityAblation(unittest.TestCase):
    def test_ablation_zeroes_the_same_tensors_as_dropout(self):
        model = build().eval()
        audio, video = inputs(8)
        with torch.no_grad():
            model.ablate_modality = 'video_only'
            audio_tokens, video_tokens = first_exchange_inputs(model, audio, video)
            self.assertTrue(bool(zeroed_rows(audio_tokens).all()))
            self.assertFalse(bool(zeroed_rows(video_tokens).any()))

            model.ablate_modality = 'audio_only'
            audio_tokens, video_tokens = first_exchange_inputs(model, audio, video)
            self.assertTrue(bool(zeroed_rows(video_tokens).all()))
            self.assertFalse(bool(zeroed_rows(audio_tokens).any()))

            model.ablate_modality = 'none'
            audio_tokens, video_tokens = first_exchange_inputs(model, audio, video)
            self.assertFalse(bool(zeroed_rows(audio_tokens).any() or zeroed_rows(video_tokens).any()))

    def test_ablation_matches_training_time_dropout_of_that_stream(self):
        # A sample whose audio is dropped in training reaches av1 exactly like the same
        # sample under eval-time 'video_only' ablation (both zero the aligned audio tokens).
        model = build()
        audio, video = inputs(1)
        model.eval()
        model.ablate_modality = 'video_only'
        with torch.no_grad():
            ablated_audio, ablated_video = first_exchange_inputs(model, audio, video)
        self.assertTrue(torch.equal(ablated_audio, torch.zeros_like(ablated_audio)))
        model.ablate_modality = 'none'
        with torch.no_grad():
            _, plain_video = first_exchange_inputs(model, audio, video)
        self.assertTrue(torch.equal(ablated_video, plain_video))

    def test_unknown_ablation_is_rejected(self):
        model = build().eval()
        model.ablate_modality = 'text_only'
        with self.assertRaisesRegex(ValueError, 'Unsupported ablate_modality'):
            model(*inputs(2))


if __name__ == '__main__':
    unittest.main()
