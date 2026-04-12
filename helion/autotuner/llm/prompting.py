"""Build prompts for the LLM-guided autotuner."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

from ..._compat import get_device_name
from ..._compat import num_compute_units
from .configs import describe_config_space
from .feedback import MAX_CHANGED_FIELDS_PER_CONFIG
from .feedback import format_config_for_prompt

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..base_search import _AutotunableKernel
    from ..config_spec import ConfigSpec

RETURN_JSON_ONLY = 'Return minified JSON only: {"configs":[...]}'
SHAPE_RULE = (
    "Do not guess field structure: for list-valued fields, emit an explicit JSON "
    "array of the exact required length; if that length is unclear, omit the "
    "field."
)
MATMUL_TARGETS = frozenset(
    {
        torch.matmul,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
)
MATMUL_API_NAMES = frozenset({"dot", "dot_scaled"})
BATCH_MATMUL_TARGET_NAMES = frozenset({"bmm", "baddbmm"})
REDUCTION_TARGET_NAMES = frozenset({"amax", "sum", "softmax", "logsumexp"})
EXP_TARGET_NAMES = frozenset({"exp", "exp2"})


def build_system_prompt() -> str:
    """Return the global instruction block shared by every LLM request."""
    return textwrap.dedent("""\
        You are an expert GPU kernel autotuner for Helion/Triton kernels.

        Use the provided Configuration Space and Default Configuration as the source of truth for:
        - allowed field names and enum values
        - which fields are scalar vs list-valued
        - required list lengths
        - valid ranges and defaults

        Key knobs:
        - block_sizes: per-dimension tile sizes. Good families usually change them coherently. For kernels with an inner or reduction dimension, very small values there are a separate aggressive family, not the default.
        - num_warps: threads/32. 4-8 is typical; 16+ is mainly for clearly larger tiles.
        - num_stages: pipeline depth. 2-4 is common for streaming loops; 1 is safer when unsure.
        - pid_type: flat, persistent_blocked, and persistent_interleaved are distinct scheduling families when available.
        - indexing: pointer and tensor_descriptor are distinct families.
        - l2_groupings, maxnreg, num_sm_multiplier, and advanced range toggles are secondary knobs; change them selectively after choosing a coherent tiling family.

        General heuristics:
        - analyze the kernel source, input tensors, GPU hardware, and config space to infer likely optimization traits from the code itself and target hardware; if unsure, stay closer to default.
        - block_sizes and num_warps should be powers of 2 when present.
        - persistent pid_type is often worth trying when total tile count is comparable to or larger than SM count, and it may also be required for some kernels.
        - tensor_descriptor is a distinct family from pointer indexing.
        - higher num_stages and multi-buffering are more aggressive and should be used selectively.

        Output contract:
        - Return minified JSON on a single line. No markdown, code fences, comments, pretty-printing, or trailing commas.
        - Emit exactly one top-level object: {"configs":[...]} and make every config unique.
        - Do not use Python syntax or expressions such as single-quoted strings or list multiplication like ["pointer"] * 4.
        - Only specify fields you want to change; unspecified = default.
        - Use only field names and enum values that appear in the config space.
        - For list-valued fields, emit an explicit JSON array with the exact required length shown in the config space.
        - Never use a scalar as shorthand for a list-valued field, and never wrap scalar-valued fields in single-element lists.
        - If you are unsure about a field's structure, required list length, or allowed values, omit that field instead of guessing.
        - Use null not None, true/false not True/False.
        - Return ONLY minified JSON: {"configs":[...]}""")


def describe_kernel(kernel: _AutotunableKernel, args: Sequence[object]) -> str:
    """Build a description of the kernel, its inputs, and the target GPU."""
    parts: list[str] = []

    try:
        inner_kernel = getattr(kernel, "kernel", None)
        if inner_kernel is not None and hasattr(inner_kernel, "fn"):
            raw_source = inspect.getsource(inner_kernel.fn)
            source_lines = textwrap.dedent(raw_source).splitlines()
            start_idx = 0
            while start_idx < len(source_lines) and not source_lines[
                start_idx
            ].lstrip().startswith("def "):
                start_idx += 1
            kernel_source = "\n".join(source_lines[start_idx:])
        else:
            kernel_source = "# Source unavailable"
    except (OSError, TypeError):
        kernel_source = "# Source unavailable"

    parts.append(f"## Kernel Source Code\n```python\n{kernel_source}\n```")

    tensor_info: list[str] = []
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensor_info.append(
                f"  arg[{index}]: shape={list(arg.shape)}, dtype={arg.dtype}"
            )
    if tensor_info:
        parts.append("## Input Tensors\n" + "\n".join(tensor_info))

    device = kernel.env.device
    device_name = get_device_name(device) or str(device)
    hw_lines = [
        f"  Device: {device_name}",
        f"  Compute units (SMs): {num_compute_units()}",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            hw_lines.extend(
                [
                    f"  Total memory: {props.total_memory / (1024**3):.1f} GB",
                    f"  Max threads per SM: {props.max_threads_per_multi_processor}",
                ]
            )
        except Exception:
            pass
    parts.append("## GPU Hardware\n" + "\n".join(hw_lines))

    return "\n\n".join(parts)


def _target_name_parts(target: object) -> frozenset[str]:
    """Extract coarse name tokens for a traced call target."""
    parts: set[str] = set()
    for raw in (
        getattr(target, "__name__", None),
        getattr(target, "name", None),
        str(target),
    ):
        if not isinstance(raw, str):
            continue
        parts.add(raw)
        parts.update(piece for piece in raw.split(".") if piece)
    return frozenset(parts)


def detect_workload_traits(
    kernel: _AutotunableKernel | None,
    *,
    config_spec: ConfigSpec | None = None,
) -> frozenset[str]:
    """Infer coarse workload traits from compiler-traced graphs."""
    if kernel is None:
        return frozenset()

    saw_matmul = False
    saw_batched_matmul = False
    saw_reduction = bool(config_spec is not None and config_spec.reduction_loops)
    saw_exp = False

    host_function = getattr(kernel, "host_function", None)
    device_ir = getattr(host_function, "device_ir", None)
    for graph_info in getattr(device_ir, "graphs", ()):
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            name_parts = _target_name_parts(node.target)
            if node.target in MATMUL_TARGETS or name_parts & MATMUL_API_NAMES:
                saw_matmul = True
            if name_parts & BATCH_MATMUL_TARGET_NAMES:
                saw_batched_matmul = True
            if name_parts & REDUCTION_TARGET_NAMES:
                saw_reduction = True
            if name_parts & EXP_TARGET_NAMES:
                saw_exp = True

    traits: set[str] = set()
    if saw_matmul:
        traits.add("matmul")
    if saw_reduction:
        traits.add("reduction")
    if saw_matmul and saw_reduction and (saw_batched_matmul or saw_exp):
        traits.add("attention_reduction")
    return frozenset(traits)


def compute_workload_hints(
    args: Sequence[object], *, workload_traits: frozenset[str] = frozenset()
) -> str:
    """Analyze the kernel workload and produce optimization hints."""
    hints: list[str] = []
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if total_bytes > 0:
        hints.append(f"Total input data: {total_bytes / (1024**2):.1f} MB")
    if workload_traits:
        hints.append("Compiler-detected traits: " + ", ".join(sorted(workload_traits)))

    shapes = [list(tensor.shape) for tensor in tensors]
    ndims = [len(shape) for shape in shapes]
    is_2d_compatible = len(tensors) >= 2 and all(dim == 2 for dim in ndims[:2])

    if "attention_reduction" in workload_traits:
        hints.extend(
            [
                (
                    "Compiler detected matmul-family ops with reductions/"
                    "normalization; keep at least one attention/reduction-style "
                    "family with moderate streaming tiles, and reserve very "
                    "aggressive num_warps/num_stages or advanced toggles for a "
                    "minority of configs."
                ),
                (
                    "For this workload, distinct families can still stay near "
                    "moderate tiles: use scheduling or indexing changes to create "
                    "diversity too, instead of forcing every family to jump to a "
                    "much larger tile."
                ),
                (
                    "Keep most balanced configs on moderate warps and stages; "
                    "for attention-style streaming kernels, 4 warps is often the "
                    "balanced choice, while 8+ warps or 4+ stages are "
                    "exploratory unless tiles are clearly large and compile "
                    "cleanly."
                ),
                (
                    "Include at least one persistent scheduling family when "
                    "available for long streaming dimensions."
                ),
            ]
        )
        if len(tensors) >= 3 and all(len(shape) >= 2 for shape in shapes[:3]):
            seq = shapes[0][-2]
            head_dim = shapes[0][-1]
            if isinstance(seq, int) and isinstance(head_dim, int):
                mid_tile = min(seq, 64)
                inner_tile = min(head_dim, 64)
                hints.extend(
                    [
                        (
                            "Attention/reduction-style starting point: try a "
                            "family near "
                            f"[1, {mid_tile}, {inner_tile}] if that matches the "
                            "config space."
                        ),
                        (
                            "Within that starting family, include a couple of "
                            "balanced variants that keep block_sizes fixed and "
                            "vary num_stages through 2-3 before moving to much "
                            "larger tiles or higher warps."
                        ),
                    ]
                )
                if inner_tile >= 32:
                    hints.append(
                        "A nearby family like "
                        f"[1, {mid_tile}, {max(16, inner_tile // 2)}] is already "
                        "distinct for this shape; prefer that kind of inner-tile "
                        "change before doubling the streaming tile, and keep tiles "
                        f"above {mid_tile} on the streaming axis to at most a "
                        "small minority."
                    )
    elif "matmul" in workload_traits and not is_2d_compatible:
        hints.append(
            "Compiler detected matmul-family ops; keep at least one coherent "
            "matmul-style tiling family in the search."
        )

    if "matmul" in workload_traits and is_2d_compatible:
        m, k = shapes[0]
        k2, n = shapes[1]
        total_tiles_64 = (m // 64) * (n // 64)
        hints.append(
            f"Matmul-like: [{m}x{k}] @ [{k2}x{n}], ~{total_tiles_64} tiles at 64x64"
        )
        if total_tiles_64 > num_compute_units() * 4:
            hints.append(
                "Problem large enough for persistent kernels - "
                "try pid_type='persistent_blocked' with l2_groupings=8-64"
            )
        hints.extend(
            [
                (
                    f"Try block_sizes near [{min(m, 128)}, {min(n, 128)}, "
                    f"{min(k, 64)}] as starting point"
                ),
                (
                    "High-perf matmul tips: try asymmetric tiles like [64,128,64], "
                    "num_stages=3-4, maxnreg=128 or 256, "
                    "range_multi_buffers=[true,true] for double-buffering, "
                    "load_eviction_policies with 'first' or 'last'"
                ),
            ]
        )
    if not hints:
        return ""
    return "\n\n## Workload Analysis\n" + "\n".join(f"  {hint}" for hint in hints)


def build_initial_search_guidance(
    *,
    configs_per_round: int,
    compile_timeout_s: int | None,
    flat_fields: Mapping[str, object],
) -> str:
    """Build the search-strategy section of the initial prompt."""
    lines = [
        (
            f"Generate up to {configs_per_round} UNIQUE candidate configs. "
            "Fewer is better than invalid JSON."
        ),
        "First analyze the kernel source, input tensors, GPU hardware, and config space.",
        "Cover 3 config families with a rough mix of about 40% near-default safe, 40% balanced throughput, and 20% aggressive configs, while keeping most candidates valid and compilable.",
        "If the kernel structure is unclear, stay closer to default and avoid aggressive coupled changes.",
        (
            "Keep each config sparse: usually 2-6 changed fields, omit unchanged defaults, "
            "and exceed 6 only when several coupled changes are needed for a distinct family."
        ),
        "Use block_sizes to define families: include at least 3 materially different tiling families instead of tiny perturbations of one tile.",
        "Vary block_sizes coherently across dimensions rather than by arbitrary skew.",
        SHAPE_RULE,
        "Do not pretty-print or repeat unchanged defaults.",
        "Avoid configs that simultaneously max out several aggressive knobs such as num_warps, num_stages, and maxnreg when present, unless strongly justified.",
    ]
    if compile_timeout_s is not None:
        lines.append(
            f"Compile timeout is {compile_timeout_s}s, so avoid candidates that are likely to compile very slowly."
        )
    if "indexing" in flat_fields:
        lines.append(
            "If tensor_descriptor is available, treat it as a separate family: include a few configs using it, but keep some pure pointer configs too."
        )
    if "pid_type" in flat_fields:
        lines.append(
            "Include both flat and persistent scheduling families when plausible; do not put every config on the same pid_type."
        )
    if "reduction_loops" in flat_fields:
        lines.append(
            "This is reduction-like: keep most configs conservative and avoid very large block_sizes or maxed num_warps/num_stages."
        )
    if any(
        name in flat_fields
        for name in (
            "range_warp_specializes",
            "range_multi_buffers",
            "range_flattens",
        )
    ):
        lines.append(
            "Use advanced toggles like warp_specialize, multi_buffer, and flatten in only a minority of otherwise sane configs."
        )
    return "## Search Strategy\n" + "\n".join(f"  - {line}" for line in lines)


def build_initial_prompt(
    *,
    kernel: _AutotunableKernel,
    args: Sequence[object],
    config_spec: ConfigSpec,
    configs_per_round: int,
    compile_timeout_s: int | None,
) -> str:
    """Build the full initial user prompt sent to the LLM."""
    default_config = config_spec.default_config()
    workload_hints = compute_workload_hints(
        args,
        workload_traits=detect_workload_traits(kernel, config_spec=config_spec),
    )
    guidance = build_initial_search_guidance(
        configs_per_round=configs_per_round,
        compile_timeout_s=compile_timeout_s,
        flat_fields=config_spec._flat_fields(),
    )
    return (
        f"{describe_kernel(kernel, args)}\n\n"
        f"## Configuration Space\n{describe_config_space(config_spec)}\n\n"
        f"## Default Configuration\n"
        f"{format_config_for_prompt(default_config)}"
        f"{workload_hints}\n\n"
        f"{guidance}\n\n"
        "## Task\n"
        "Suggest the first batch of configs. Include both near-default and exploratory candidates. "
        f"{RETURN_JSON_ONLY}"
    )


def build_refinement_prompt(
    *,
    configs_per_round: int,
    compile_timeout_s: int | None,
    failed_count: int,
    total_count: int,
    search_state: str,
    anchor_configs: str,
    results: str,
    top_patterns: str,
    failed_patterns: str,
) -> str:
    """Build the refinement prompt sent after each benchmarking round."""
    if total_count > 0 and failed_count * 3 >= total_count:
        strategy_lines = [
            "Recent rounds had many failures. Use only the best 1-2 anchors.",
            "At least 80% of configs should be 1-2 field mutations of those anchors.",
            "Back off aggressive settings first: smaller num_stages/num_warps, pointer indexing, fewer advanced toggles.",
        ]
    else:
        strategy_lines = [
            "About two thirds of configs should be 1-field mutations of Anchor 1.",
            "Use most of the rest for 1-2 field mutations of Anchor 2.",
            "Reserve at most a small minority for one clearly different family, not random noise.",
        ]
    strategy_lines.append(
        "Prefer edits with attributable effects: change block_sizes, num_warps, num_stages, pid_type, indexing, l2_groupings, or maxnreg instead of rewriting every field."
    )
    strategy_lines.append(
        "Keep each config sparse: usually 1-4 changed fields, and no more than "
        f"{MAX_CHANGED_FIELDS_PER_CONFIG} unless absolutely necessary."
    )
    strategy_lines.append(SHAPE_RULE)
    if compile_timeout_s is not None:
        strategy_lines.append(
            f"Keep compile cost in mind: avoid candidates that are likely to exceed the {compile_timeout_s}s compile timeout."
        )
    strategy_lines.append(
        "If unsure, return fewer valid configs instead of verbose or malformed JSON."
    )
    return (
        f"## Search State\n{search_state}\n\n"
        f"## Anchor Configs\n{anchor_configs}\n\n"
        f"## Results (best first)\n{results}\n\n"
        f"## Top Config Patterns\n{top_patterns}\n\n"
        f"## Failed Config Patterns\n{failed_patterns}\n\n"
        "## Next Step\n" + "\n".join(f"  - {line}" for line in strategy_lines) + "\n\n"
        "## Task\n"
        f"Suggest up to {configs_per_round} NEW UNIQUE configs around the anchors above. "
        "Avoid the failed patterns above and favor targeted edits with attributable effects. "
        f"{RETURN_JSON_ONLY}"
    )
