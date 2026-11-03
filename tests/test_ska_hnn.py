"""Unit tests for SKA-HNN model modules, attention constraints, numerics, and switches.

Verifies:
1. Output shapes per module and full model at variable T (T = 100, 180, 400).
2. Attention weights strictly sum to 1.0 over their designated normalization axes.
3. Local statistics output is torch.float32 and handles constant/zero-variance input without NaN.
4. Backward pass delivers non-zero gradients to every single learnable parameter.
5. Eval-mode batch invariance for equal-length batches (max abs diff < 1e-4).
6. Ablation switches S0, S1, S2, S3, S5, S6 build, run forward/backward, and parameter
   counts differ in the theoretically expected directions.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ska_hnn import (
    ChannelWiseSKA,
    FrequencyWiseSKA,
    FullyTDNNModule,
    LocalStatsPool1d,
    MLASP,
    MultiLevelFeatureCombination,
    SEBlock2D,
    SKAHNN,
    SKAResBlock,
    TDNNLSTMModule,
    build_ska_hnn,
)


@pytest.mark.parametrize("T", [100, 180, 400])
def test_module_output_shapes_variable_t(T: int) -> None:
    """Verify that every individual sub-module and the full model output correct shapes at variable T."""
    B = 2
    C = 32
    C1 = 256
    d_c = 640
    d_i = d_c + 2 * C1  # 1152
    emb_dim = 192

    # 1. LocalStatsPool1d
    ls = LocalStatsPool1d(window_size=21, eps=1e-5, concat_input=True)
    x_seq = torch.randn(B, d_c, T)
    out_ls = ls(x_seq)
    assert out_ls.shape == (B, 3 * d_c, T), f"LS shape mismatch at T={T}: {out_ls.shape}"

    # 2. fwSKA
    fw = FrequencyWiseSKA(channels=C, freq_bins=40, reduction=8)
    x_2d = torch.randn(B, C, 40, T)
    out_fw = fw(x_2d)
    assert out_fw.shape == (B, C, 40, T), f"fwSKA shape mismatch at T={T}: {out_fw.shape}"

    # 3. cwSKA
    cw = ChannelWiseSKA(channels=C, reduction=8)
    out_cw = cw(x_2d)
    assert out_cw.shape == (B, C, 40, T), f"cwSKA shape mismatch at T={T}: {out_cw.shape}"

    # 4. SEBlock2D
    se = SEBlock2D(channels=C, reduction=8)
    out_se = se(x_2d)
    assert out_se.shape == (B, C, 40, T), f"SE shape mismatch at T={T}: {out_se.shape}"

    # 5. SKAResBlock
    resblock = SKAResBlock(channels=C, freq_bins=40, reduction=8, use_ska=True)
    out_res = resblock(x_2d)
    assert out_res.shape == (B, C, 40, T), f"ResBlock shape mismatch at T={T}: {out_res.shape}"

    # 6. TDNNLSTMModule
    tdnn_lstm = TDNNLSTMModule(in_dim=3 * d_c, out_dim=C1)
    x_tdnn_in = torch.randn(B, 3 * d_c, T)
    out_lstm = tdnn_lstm(x_tdnn_in)
    assert out_lstm.shape == (B, C1, T), f"TDNN-LSTM shape mismatch at T={T}: {out_lstm.shape}"

    # 7. FullyTDNNModule
    fully_tdnn = FullyTDNNModule(in_dim=3 * C1, out_dim=C1)
    x_fully_in = torch.randn(B, 3 * C1, T)
    out_fully = fully_tdnn(x_fully_in)
    assert out_fully.shape == (B, C1, T), f"Fully-TDNN shape mismatch at T={T}: {out_fully.shape}"

    # 8. MultiLevelFeatureCombination
    mlfc = MultiLevelFeatureCombination(use_mlfc=True)
    out_mlfc = mlfc(x_t=out_fully, x_l=out_lstm, x_c=torch.randn(B, d_c, T))
    assert out_mlfc.shape == (B, d_i, T), f"MLFC shape mismatch at T={T}: {out_mlfc.shape}"

    # 9. MLASP
    mlasp = MLASP(feat_dim=d_i, attn_dim=128, use_ls_context=True, window_size=21)
    out_mlasp = mlasp(out_mlfc)
    assert out_mlasp.shape == (B, 2 * d_i), f"MLASP shape mismatch at T={T}: {out_mlasp.shape}"

    # 10. Full SKAHNN Model (both training and eval modes)
    model = build_ska_hnn(switch="S0", c_cnn=C, c_tdnn=C1, emb_dim=emb_dim)
    x_input = torch.randn(B, 80, T)

    model.train()
    emb_train = model(x_input)
    assert emb_train.shape == (B, emb_dim), f"Full model train shape mismatch at T={T}: {emb_train.shape}"

    model.eval()
    with torch.no_grad():
        emb_eval = model(x_input)
    assert emb_eval.shape == (B, emb_dim), f"Full model eval shape mismatch at T={T}: {emb_eval.shape}"


def test_attention_weights_sum_to_one() -> None:
    """Verify that attention weights sum to 1.0 along their designated normalization axes."""
    B = 3
    C = 32
    F_bins = 40
    T = 180
    d_i = 1152

    # 1. fwSKA: attention weights over the 2 branches for each frequency bin (dim=1)
    fw = FrequencyWiseSKA(channels=C, freq_bins=F_bins, reduction=8)
    x_fw = torch.randn(B, C, F_bins, T)
    _, attn_fw = fw(x_fw, return_attn=True)  # (B, 2, F_bins)
    assert attn_fw.shape == (B, 2, F_bins)
    sum_fw = attn_fw.sum(dim=1)  # sum over branch dim=1
    assert torch.allclose(sum_fw, torch.ones_like(sum_fw), atol=1e-6), (
        f"fwSKA attention does not sum to 1 across branches: max diff={torch.max(torch.abs(sum_fw - 1.0)).item()}"
    )

    # 2. cwSKA: attention weights over the 2 branches for each channel (dim=1)
    cw = ChannelWiseSKA(channels=C, reduction=8)
    x_cw = torch.randn(B, C, F_bins, T)
    _, attn_cw = cw(x_cw, return_attn=True)  # (B, 2, C)
    assert attn_cw.shape == (B, 2, C)
    sum_cw = attn_cw.sum(dim=1)  # sum over branch dim=1
    assert torch.allclose(sum_cw, torch.ones_like(sum_cw), atol=1e-6), (
        f"cwSKA attention does not sum to 1 across branches: max diff={torch.max(torch.abs(sum_cw - 1.0)).item()}"
    )

    # 3. MLASP: attention weights over the temporal frames T for each channel (dim=2)
    mlasp = MLASP(feat_dim=d_i, attn_dim=128, use_ls_context=True, window_size=21)
    x_mlasp = torch.randn(B, d_i, T)
    _, alpha = mlasp(x_mlasp, return_attn=True)  # (B, d_i, T)
    assert alpha.shape == (B, d_i, T)
    sum_alpha = alpha.sum(dim=2)  # sum over temporal dim=2
    assert torch.allclose(sum_alpha, torch.ones_like(sum_alpha), atol=1e-6), (
        f"MLASP attention does not sum to 1 across frames: max diff={torch.max(torch.abs(sum_alpha - 1.0)).item()}"
    )


def test_local_stats_fp32_and_constant_input() -> None:
    """Verify local statistics numerics: fp32 output and zero-variance guard against NaN."""
    ls = LocalStatsPool1d(window_size=21, eps=1e-5, concat_input=True)

    # Test 1: Explicit fp32 output when input is float16 or float32
    for dtype in [torch.float32, torch.float16]:
        x = torch.randn(2, 64, 180, dtype=dtype)
        out = ls(x)
        assert out.dtype == torch.float32, f"Expected output in torch.float32, got {out.dtype} for input {dtype}"
        assert not torch.isnan(out).any(), f"NaN encountered with input dtype {dtype}"

    # Test 2: Constant input (zero temporal variance)
    constant_val = 5.739
    x_const = torch.full((4, 32, 150), fill_value=constant_val, dtype=torch.float32)
    out_const = ls(x_const)

    # out_const is [X; mean; std]
    mean_part = out_const[:, 32:64, :]
    std_part = out_const[:, 64:96, :]

    assert not torch.isnan(out_const).any(), "NaN found in constant input local stats!"
    assert not torch.isinf(out_const).any(), "Inf found in constant input local stats!"
    assert torch.allclose(mean_part, x_const, atol=1e-4), "Mean does not equal constant input value"
    # Std should equal sqrt(eps) = sqrt(1e-5) ~= 0.003162
    expected_std = 1e-5 ** 0.5
    assert torch.allclose(std_part, torch.full_like(std_part, expected_std), atol=1e-4), (
        f"Std on constant input should clamp to sqrt(eps)={expected_std}, got max {std_part.max().item()}"
    )


def test_gradients_reach_every_parameter() -> None:
    """Verify that a single backward pass produces non-zero gradients for every learnable parameter."""
    torch.manual_seed(42)
    model = build_ska_hnn(switch="S0", c_cnn=32, c_tdnn=256, emb_dim=192)
    model.train()

    x = torch.randn(4, 80, 180, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    missing_grads = []
    zero_grads = []
    nan_grads = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing_grads.append(name)
        elif torch.isnan(param.grad).any():
            nan_grads.append(name)
        elif param.grad.abs().sum().item() == 0.0:
            zero_grads.append(name)

    assert not missing_grads, f"Parameters with missing gradients: {missing_grads}"
    assert not nan_grads, f"Parameters with NaN gradients: {nan_grads}"
    assert not zero_grads, f"Parameters with all-zero gradients: {zero_grads}"


def test_eval_mode_batch_invariance() -> None:
    """Verify that eval-mode embeddings for an utterance are identical whether extracted solo or in a batch."""
    torch.manual_seed(42)
    model = build_ska_hnn(switch="S0", c_cnn=32, c_tdnn=256, emb_dim=192)
    model.eval()

    u1 = torch.randn(1, 80, 180)
    u2 = torch.randn(1, 80, 180)
    u_batch = torch.cat([u1, u2], dim=0)  # (2, 80, 180)

    with torch.no_grad():
        emb1 = model(u1)
        emb2 = model(u2)
        emb_batch = model(u_batch)

    diff1 = torch.max(torch.abs(emb_batch[0:1] - emb1)).item()
    diff2 = torch.max(torch.abs(emb_batch[1:2] - emb2)).item()

    assert diff1 < 1e-4, f"Batch invariance violated for sample 1: max abs diff={diff1}"
    assert diff2 < 1e-4, f"Batch invariance violated for sample 2: max abs diff={diff2}"


def test_ablation_switches_build_run_and_parameter_delta() -> None:
    """Verify that each switch S0, S1, S2, S3, S5, S6 builds, runs forward/backward, and parameter deltas match expectations."""
    switches = ["S0", "S1", "S2", "S3", "S5", "S6"]
    param_counts = {}

    for sw in switches:
        torch.manual_seed(42)
        model = build_ska_hnn(switch=sw, c_cnn=32, c_tdnn=256, emb_dim=192)
        model.train()

        total_params = sum(p.numel() for p in model.parameters())
        param_counts[sw] = total_params

        # Run forward pass
        x = torch.randn(2, 80, 180, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 192), f"Switch {sw} returned unexpected output shape {out.shape}"

        # Run backward pass
        loss = out.sum()
        loss.backward()

        # Assert no NaNs in gradients
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in switch {sw}"

    # Verify parameter relationships:
    # 1. S0 and S1 have identical model architectures (only loss differs in training)
    assert param_counts["S0"] == param_counts["S1"], (
        f"S0 ({param_counts['S0']}) != S1 ({param_counts['S1']})"
    )

    # 2. S2 removes MLASP context (attention in_dim = d_i instead of 3*d_i) -> fewer params than S1
    assert param_counts["S2"] < param_counts["S1"], (
        f"Expected S2 ({param_counts['S2']}) < S1 ({param_counts['S1']})"
    )

    # 3. S3 removes MLFC (d_i = c_tdnn = 256 instead of 1152) -> fewer params than S1
    assert param_counts["S3"] < param_counts["S1"], (
        f"Expected S3 ({param_counts['S3']}) < S1 ({param_counts['S1']})"
    )

    # 4. S5 replaces fwSKA and cwSKA with vanilla Conv2d -> fewer params than S1
    assert param_counts["S5"] < param_counts["S1"], (
        f"Expected S5 ({param_counts['S5']}) < S1 ({param_counts['S1']})"
    )

    # 5. S6 removes 2D-CNN frontend entirely -> significantly fewer params than S1
    assert param_counts["S6"] < param_counts["S1"], (
        f"Expected S6 ({param_counts['S6']}) < S1 ({param_counts['S1']})"
    )


def test_extractor_variable_length_and_zero_padding_parity() -> None:
    """Verify that EmbeddingExtractor with SKA-HNN handles variable-length utterances correctly.
    
    1. Tests that extract_batch_bucketed on variable-length audio matches single-waveform
       extraction (alone, batch_size=1) within < 1e-4 tolerance.
    2. Demonstrates that naive zero-padding distorts embeddings (diff >> 1e-4), validating
       the extractor's design choice of zero-padding avoidance via length bucketing.
    """
    from src.eval.extractor import EmbeddingExtractor

    torch.manual_seed(42)
    model = build_ska_hnn("S0", c_cnn=32, c_tdnn=256, emb_dim=192, attn_dim=128)
    extractor = EmbeddingExtractor(model, device=torch.device("cpu"))

    # 3 distinct utterance lengths: 1.0s (16k), 1.5s (24k), 2.0s (32k)
    w1 = torch.randn(16000)
    w2 = torch.randn(24000)
    w3 = torch.randn(32000)

    # Individual extractions (alone, batch size 1)
    e1_alone = extractor.extract_from_waveform(w1)
    e2_alone = extractor.extract_from_waveform(w2)
    e3_alone = extractor.extract_from_waveform(w3)

    # Bucketed batch extraction containing variable lengths and repeats
    embs_bucketed = extractor.extract_batch_bucketed([w1, w2, w3, w1, w2])

    diff1 = np.max(np.abs(e1_alone - embs_bucketed[0]))
    diff2 = np.max(np.abs(e2_alone - embs_bucketed[1]))
    diff3 = np.max(np.abs(e3_alone - embs_bucketed[2]))
    diff1_rep = np.max(np.abs(e1_alone - embs_bucketed[3]))
    diff2_rep = np.max(np.abs(e2_alone - embs_bucketed[4]))

    # Must be virtually bitwise identical (< 1e-4 required, typically < 1e-6)
    assert max(diff1, diff2, diff3, diff1_rep, diff2_rep) < 1e-4, (
        f"Bucketed extraction deviated from alone: max diff={max(diff1, diff2, diff3, diff1_rep, diff2_rep):.2e}"
    )

    # Demonstrate that naive zero-padding distorts the embedding
    w1_padded = torch.zeros(32000)
    w1_padded[:16000] = w1
    e1_padded = extractor.extract_from_waveform(w1_padded)
    diff_padded = np.max(np.abs(e1_alone - e1_padded))
    assert diff_padded > 1e-2, (
        f"Expected zero-padding to alter the embedding, but diff was unexpectedly small ({diff_padded:.2e})"
    )


