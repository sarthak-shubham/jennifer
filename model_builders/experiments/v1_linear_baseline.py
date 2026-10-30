"""
Colony: One - Iteration 1: Linear Policy-Value Baseline Model
Opset 18, IR version 9, dynamic batching.

Ablation baseline evaluating material weight matrices and center-ring bonuses
without spatial convolution. Kept for benchmarking and architectural comparison.
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def build_linear_baseline(output_path: str = None) -> Path:
    # 1. Value Head Weights (16, 8, 8)
    val_weights = np.zeros((16, 8, 8), dtype=np.float32)

    # Material values:
    # Friendly (Planes 0..5)
    val_weights[0] += 0.4   # Ping
    val_weights[1] += 0.9   # Daemon
    val_weights[2] += 0.5   # Exploit
    val_weights[3] += 0.5   # Trace
    val_weights[4] += 0.7   # Ice
    val_weights[5] += 10.0  # Kernel

    # Enemy (Planes 6..11)
    val_weights[6] -= 0.4   # Enemy Ping
    val_weights[7] -= 0.9   # Enemy Daemon
    val_weights[8] -= 0.5   # Enemy Exploit
    val_weights[9] -= 0.5   # Enemy Trace
    val_weights[10] -= 0.7  # Enemy Ice
    val_weights[11] -= 10.0 # Enemy Kernel

    # Positional bonus (center 4x4)
    for r in range(8):
        for c in range(8):
            ring = max(abs(r - 3.5), abs(c - 3.5)) - 0.5
            center_val = (3.0 - ring) * 0.1
            for p in range(6):
                val_weights[p, r, c] += center_val
            for p in range(6, 12):
                val_weights[p, r, c] -= center_val

    val_w_flat = val_weights.reshape(1024, 1)

    # 2. Policy Head Weights
    to_weights = np.zeros((16, 8, 8), dtype=np.float32)
    to_weights[11] += 15.0  # Enemy Kernel
    to_weights[7] += 6.0    # Enemy Daemon
    to_weights[10] += 5.0   # Enemy Ice
    to_weights[8] += 4.0    # Enemy Exploit
    to_weights[9] += 4.0    # Enemy Trace
    to_weights[6] += 3.5    # Enemy Ping

    for r in range(8):
        for c in range(8):
            ring = max(abs(r - 3.5), abs(c - 3.5)) - 0.5
            center_bias = (3.0 - ring) * 0.5
            for p in range(16):
                to_weights[p, r, c] += center_bias / 16.0

    to_w_per_sq = to_weights.reshape(16, 64)

    from_weights = np.zeros((16, 8, 8), dtype=np.float32)
    for p in range(5):
        from_weights[p] += 1.0
    for r in range(8):
        for c in range(8):
            if r in (0, 7) or c in (0, 7):
                for p in range(6):
                    from_weights[p, r, c] += 1.0

    from_w_per_sq = from_weights.reshape(16, 64)

    # 3. ONNX Graph Definition
    obs_info = helper.make_tensor_value_info('observation', TensorProto.FLOAT, ['batch', 16, 8, 8])
    policy_info = helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 4096])
    value_info = helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch'])

    nodes = []
    initializers = []

    shape_1024 = helper.make_tensor('shape_1024', TensorProto.INT64, [2], [-1, 1024])
    initializers.append(shape_1024)
    nodes.append(helper.make_node('Reshape', ['observation', 'shape_1024'], ['obs_flat']))

    # Value Head: MatMul -> Tanh -> Squeeze
    w_val_tensor = helper.make_tensor('W_val', TensorProto.FLOAT, [1024, 1], val_w_flat.flatten().tolist())
    initializers.append(w_val_tensor)
    nodes.append(helper.make_node('MatMul', ['obs_flat', 'W_val'], ['val_raw']))
    nodes.append(helper.make_node('Tanh', ['val_raw'], ['val_tanh']))

    axes_1 = helper.make_tensor('axes_1', TensorProto.INT64, [1], [1])
    initializers.append(axes_1)
    nodes.append(helper.make_node('Squeeze', ['val_tanh', 'axes_1'], ['value']))

    # Policy Head: Factorized (from_score + to_score)
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
        name='ColonyOneLinearBaseline',
        inputs=[obs_info],
        outputs=[policy_info, value_info],
        initializer=initializers
    )

    if output_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        out_file = repo_root / "models" / "experiments" / "v1_linear_baseline.onnx"
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
    print(f"Successfully generated Linear Baseline model:")
    print(f"  Target:     {out_file}")
    print(f"  Size:       {size_kb:.2f} KB")
    print(f"  Opset / IR: Opset 18 (IR Version 9)")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and export the Colony: One Linear Baseline (Iteration 1) ONNX model."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Custom destination path for the exported .onnx model."
    )
    args = parser.parse_args()
    build_linear_baseline(args.output)


if __name__ == '__main__':
    main()
