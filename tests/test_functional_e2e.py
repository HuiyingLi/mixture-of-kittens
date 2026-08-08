import itertools

import torch
import torch.distributed as dist

from mok import functional
from mok.functional import get_workspace
from mok.ops import mxfp8_quantize

from .utils import (
    check_correctness,
    generate_inputs,
    mok_params,
    run_reference_bf16,
    shapes,
)

BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
RESULT_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


def test_e2e_bf16(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, params in itertools.product(shapes(world_size), mok_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        params_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = params
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        inputs = generate_inputs(
            rank,
            device,
            num_experts,
            num_local_experts,
            topk,
            num_local_tokens,
            hidden_dim,
            intermediate_dim,
        )
        (
            x,
            topk_experts,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            d_output,
        ) = inputs

        reference_results = run_reference_bf16(*inputs)
        config = functional.MoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=1.5,  # must be lower for production
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )

        bf16_output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
        )
        bf16_gradients = functional.backward(
            config,
            workspace,
            schedule,
            forward_context,
            d_output,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
        )
        bf16_results = (bf16_output, *bf16_gradients)

        for name, reference, actual in zip(RESULT_NAMES, reference_results, bf16_results, strict=True):
            check_correctness(
                f"{shape_name}/{params_name}/{name}",
                reference,
                actual,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )


def test_e2e_bf16_shared_output_gate(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    inputs = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    (
        x,
        topk_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    ) = inputs
    generator = torch.Generator(device=device).manual_seed(4321 + rank)
    shared_output_gate = torch.sigmoid(
        torch.randn((num_local_tokens, 1), device=device, dtype=torch.bfloat16, generator=generator)
    )
    reference_results = run_reference_bf16(*inputs, shared_output_gate=shared_output_gate)
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=1.5,
        all_gather_top_experts_chunk_bytes=16,
    )
    workspace = get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )

    output, _, forward_context = functional.forward_with_shared_output(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        shared_output_gate,
    )
    gradients = functional.backward(
        config,
        workspace,
        schedule,
        forward_context,
        d_output,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        shared_grad_output=d_output * shared_output_gate,
    )

    for name, reference, actual in zip(
        RESULT_NAMES,
        reference_results,
        (output, *gradients),
        strict=True,
    ):
        check_correctness(
            f"shared output gate/{name}",
            reference,
            actual,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )


def test_e2e_mxfp8(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, params in itertools.product(shapes(world_size), mok_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        params_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = params
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        inputs = generate_inputs(
            rank,
            device,
            num_experts,
            num_local_experts,
            topk,
            num_local_tokens,
            hidden_dim,
            intermediate_dim,
        )
        (
            x,
            topk_experts,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            d_output,
        ) = inputs

        reference_results = run_reference_bf16(*inputs)
        config = functional.MoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=1.5,  # must be lower for production
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )

        (
            w_routed_gate_fp8,
            w_routed_gate_sc,
            w_routed_gate_t_fp8,
            w_routed_gate_t_sc,
        ) = mxfp8_quantize(w_routed_gate, True, True)
        (
            w_routed_up_fp8,
            w_routed_up_sc,
            w_routed_up_t_fp8,
            w_routed_up_t_sc,
        ) = mxfp8_quantize(w_routed_up, True, True)
        (
            w_routed_down_fp8,
            w_routed_down_sc,
            w_routed_down_t_fp8,
            w_routed_down_t_sc,
        ) = mxfp8_quantize(w_routed_down, True, True)

        mxfp8_output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            (w_routed_gate_fp8, w_routed_gate_sc),
            (w_routed_up_fp8, w_routed_up_sc),
            (w_routed_down_fp8, w_routed_down_sc),
        )
        mxfp8_gradients = functional.backward(
            config,
            workspace,
            schedule,
            forward_context,
            d_output,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            (
                w_routed_gate_fp8,
                w_routed_gate_sc,
                w_routed_gate_t_fp8,
                w_routed_gate_t_sc,
            ),
            (
                w_routed_up_fp8,
                w_routed_up_sc,
                w_routed_up_t_fp8,
                w_routed_up_t_sc,
            ),
            (
                w_routed_down_t_fp8,
                w_routed_down_t_sc,
            ),
        )
        mxfp8_results = (mxfp8_output, *mxfp8_gradients)

        for name, reference, actual in zip(RESULT_NAMES, reference_results, mxfp8_results, strict=True):
            check_correctness(
                f"{shape_name}/{params_name}/{name}",
                reference,
                actual,
                MXFP8_TOLERANCE,
                print_stats=rank == 0,
            )
