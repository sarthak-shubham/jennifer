"""
Colony: One - Iteration 2: Feature Broadcaster Model
Opset 18, IR version 9, dynamic batching.

Ablation baseline evaluating tactical piece weighting, firewall evacuation,
and broadcasting mechanisms without spatial convolution. Kept for benchmarking.
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def build_feature_broadcaster(output_path: str = None) -> Path:
    # 1. Value Head Weights (16, 8, 8)
    val_w = np.zeros((16, 8, 8), dtype=np.float32)

    p_vals = [0.6, 1.8, 1.0, 1.2, 2.0, 15.0]

    for p in range(6):
        val_w[p] += p_vals[p]
        val_w[p + 6] -= p_vals[p]

    for r in range(8):
        for c in range(8):
            ring = max(abs(r - 3.5), abs(c - 3.5)) - 0.5
            center_bonus = (1.5 - ring) * 0.3
            for p in range(6):
                val_w[p, r, c] += center_bonus
                val_w[p + 6, r, c] -= center_bonus

    # Plane 14: Firewall urgency penalty for friendly pieces on outer ring
    for r in range(8):
        for c in range(8):
            if r in (0, 7) or c in (0, 7):
                val_w[14, r, c] -= 0.5

    val_w_flat = val_w.reshape(1024, 1)

    # 2. Policy Head Weights (4096 = 64 * 64)
    to_w = np.zeros((16, 8, 8), dtype=np.float32)
    to_w[11] += 25.0  # Enemy Kernel capture
    to_w[7] += 12.0   # Enemy Daemon capture
    to_w[10] += 10.0  # Enemy Ice capture
    to_w[8] += 7.0    # Enemy Exploit capture
    to_w[9] += 7.0    # Enemy Trace capture
    to_w[6] += 5.0    # Enemy Ping capture

    for r in range(8):
        for c in range(8):
            ring = max(abs(r - 3.5), abs(c - 3.5)) - 0.5
            bias = (1.5 - ring) * 0.8
            for p in range(16):
                to_w[p, r, c] += bias / 16.0

    to_w_per_sq = to_w.reshape(16, 64)

    from_w = np.zeros((16, 8, 8), dtype=np.float32)
    for p in [1, 4, 2, 3, 0]:
        from_w[p] += 1.5
    from_w[5] += 0.5

    for r in range(8):
        for c in range(8):
            if r in (0, 7) or c in (0, 7):
                for p in range(6):
                    from_w[p, r, c] += 2.0

    from_w_per_sq = from_w.reshape(16, 64)

    # 3. ONNX Graph Definition
    obs_info = helper.make_tensor_value_info('observation', TensorProto.FLOAT, ['batch', 16, 8, 8])
    policy_info = helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 4096])
    value_info = helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch'])

    nodes = []
    initializers = []

    shape_1024 = helper.make_tensor('shape_1024', TensorProto.INT64, [2], [-1, 1024])
    initializers.append(shape_1024)
    nodes.append(helper.make_node('Reshape', ['observation', 'shape_1024'], ['obs_flat']))

    w_val_tensor = helper.make_tensor('W_val', TensorProto.FLOAT, [1024, 1], val_w_flat.flatten().tolist())
    initializers.append(w_val_tensor)
    nodes.append(helper.make_node('MatMul', ['obs_flat', 'W_val'], ['val_raw']))
    nodes.append(helper.make_node('Tanh', ['val_raw'], ['val_tanh']))

    axes_1 = helper.make_tensor('axes_1', TensorProto.INT64, [1], [1])
    initializers.append(axes_1)
    nodes.append(helper.make_node('Squeeze', ['val_tanh', 'axes_1'], ['value']))

    shape_16_64 = helper.make_tensor('shape_16_64', TensorProto.INT64, [3], [-1, 16, 64])
    initializers.append(shape_16_64)
    nodes.append(helper.make_node('Reshape', ['observation', 'shape_16_64'], ['obs_16_64']))

    w_to_tensor = helper.make_tensor('W_to', TensorProto.FLOAT, [1, 16, 64], to_w_per_sq.tolist())
    initializers.append(w_to_tensor)
    nodes.append(helper.make_node('Mul', ['obs_16_64', 'W_to'], ['to_elem_mul']))

    channel_axis = helper.make_tensor('channel_axis', TensorProto.INT64, [1], [1])
    initializers.append(channel_axis)
    nodes.append(helper.make_node('ReduceSum', ['to_elem_mul', 'channel_axis'], ['to_score'], keepdims=1))

    w_from_tensor = helper.make_tensor('W_from', TensorProto.FLOAT, [1, 16, 64], from_w_per_sq.tolist())
    initializers.append(w_from_tensor)
    nodes.append(helper.make_node('Mul', ['obs_16_64', 'W_from'], ['from_elem_mul']))
    nodes.append(helper.make_node('ReduceSum', ['from_elem_mul', 'channel_axis'], ['from_score_1d'], keepdims=1))
    nodes.append(helper.make_node('Transpose', ['from_score_1d'], ['from_score'], perm=[0, 2, 1]))

    nodes.append(helper.make_node('Add', ['from_score', 'to_score'], ['grid_scores']))
    shape_4096 = helper.make_tensor('shape_4096', TensorProto.INT64, [2], [-1, 4096])
    initializers.append(shape_4096)
    nodes.append(helper.make_node('Reshape', ['grid_scores', 'shape_4096'], ['policy']))

    graph = helper.make_graph(
        nodes=nodes,
        name='ColonyOneFeatureBroadcaster',
        inputs=[obs_info],
        outputs=[policy_info, value_info],
        initializer=initializers
    )

    if output_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        out_file = repo_root / "models" / "experiments" / "v2_feature_broadcaster.onnx"
    else:
        out_file = Path(output_path).resolve()

    out_file.parent.mkdir(parents=True, exist_ok=True)

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 18)],
        ir_version=9
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(out_file))

    size_kb = out_file.stat().st_size / 1024.0
    print(f"Successfully generated Feature Broadcaster model:")
    print(f"  Target:     {out_file}")
    print(f"  Size:       {size_kb:.2f} KB")
    print(f"  Opset / IR: Opset 18 (IR Version 9)")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and export the Colony: One Feature Broadcaster (Iteration 2) ONNX model."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Custom destination path for the exported .onnx model."
    )
    args = parser.parse_args()
    build_feature_broadcaster(args.output)


if __name__ == '__main__':
    main()
