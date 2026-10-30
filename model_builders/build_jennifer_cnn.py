"""
Colony: One - Jennifer Tactical CNN Model Builder
Compiles an ONNX model (Opset 18, IR version 9) with dynamic batching.

Architectural Features:
1. 3x3 Spatial Convolution:
   - Filter 0: Friendly Bodyguard escort detection at Kernel position.
   - Filter 1: Enemy Bodyguard escort detection at Enemy Kernel.
   - Filter 2: Enemy threat zone (Daemon adjacent shoot + Kernel attack).
   - Filter 3: Friendly Daemon kill zone.
   - Filter 4: Local material dominance in 3x3 neighborhood.
   - Filter 5: Center control & firewall safety ring.
   - Filter 6: Action 1 ambush setup (modulated by plane 13).
   - Filter 7: Threat to friendly Kernel.
2. Value Head: Tanh linear projection of concatenated raw + conv feature tensor.
3. Policy Head: Factorized from_score + to_score matrix outer-sum over (64 x 64).
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def build_jennifer_cnn(output_path: str = None) -> Path:
    # 1. Convolution Layer Filters (C_out=8, C_in=16, K=3, K=3)
    conv_w = np.zeros((8, 16, 3, 3), dtype=np.float32)
    conv_b = np.zeros(8, dtype=np.float32)

    # Filter 0: Friendly Bodyguard Detection at Kernel position
    conv_w[0, 5, 1, 1] = 1.0
    for p in range(5):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                conv_w[0, p, 1 + dr, 1 + dc] = 0.25

    # Filter 1: Enemy Bodyguard Detection at Enemy Kernel position
    conv_w[1, 11, 1, 1] = 1.0
    for p in range(6, 11):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                conv_w[1, p, 1 + dr, 1 + dc] = 0.25

    # Filter 2: Enemy Threat Zone (Daemon adjacent shoot + Kernel adjacent attack)
    for p in [7, 11]:
        weight = 1.0 if p == 7 else 0.5
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                conv_w[2, p, 1 + dr, 1 + dc] = weight

    # Filter 3: Friendly Daemon Kill Zone
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            conv_w[3, 1, 1 + dr, 1 + dc] = 1.0

    # Filter 4: Local Material Dominance in 3x3
    p_weights = [0.5, 1.5, 1.0, 1.0, 1.5, 5.0]
    for p in range(6):
        conv_w[4, p, 1, 1] = p_weights[p]
        conv_w[4, p + 6, 1, 1] = -p_weights[p]

    # Filter 5: Center Control & Firewall Safe Ring
    for r in range(3):
        for c in range(3):
            conv_w[5, 5, r, c] = 0.2

    # Filter 6: Action 1 Ambush Setup (Plane 13)
    conv_w[6, 13, 1, 1] = 1.0

    # Filter 7: Threat to Friendly Kernel
    for p in range(6, 12):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                conv_w[7, p, 1 + dr, 1 + dc] = 0.5

    # 2. Value Head Weights
    val_raw_w = np.zeros((16, 8, 8), dtype=np.float32)
    p_vals = [0.6, 2.0, 1.0, 1.2, 2.0, 20.0]
    for p in range(6):
        val_raw_w[p] += p_vals[p]
        val_raw_w[p + 6] -= p_vals[p]

    for r in range(8):
        for c in range(8):
            ring = max(abs(r - 3.5), abs(c - 3.5)) - 0.5
            bonus = (1.5 - ring) * 0.25
            for p in range(6):
                val_raw_w[p, r, c] += bonus
                val_raw_w[p + 6, r, c] -= bonus

    val_conv_w = np.zeros((8, 8, 8), dtype=np.float32)
    val_conv_w[0] += 1.5
    val_conv_w[1] -= 2.0
    val_conv_w[3] += 0.8
    val_conv_w[7] -= 3.0

    val_w_total = np.concatenate([val_raw_w.flatten(), val_conv_w.flatten()]).reshape(-1, 1)

    # 3. Policy Head Weights
    to_w_3d = np.zeros((16, 8, 8), dtype=np.float32)
    to_w_3d[11] += 30.0  # Enemy Kernel capture
    to_w_3d[7] += 15.0   # Enemy Daemon capture
    to_w_3d[10] += 12.0  # Enemy Ice capture
    to_w_3d[8] += 8.0    # Enemy Exploit capture
    to_w_3d[9] += 8.0    # Enemy Trace capture
    to_w_3d[6] += 6.0    # Enemy Ping capture

    for r in range(8):
        for c in range(8):
            ring = max(abs(r - 3.5), abs(c - 3.5)) - 0.5
            bias = (1.5 - ring) * 0.7
            for p in range(16):
                to_w_3d[p, r, c] += bias / 16.0
    to_raw_w = to_w_3d.reshape(16, 64)

    to_conv_3d = np.zeros((8, 8, 8), dtype=np.float32)
    to_conv_3d[2] -= 6.0  # Penalize moves into enemy threat zones
    to_conv_3d[3] += 2.0  # Bonus for friendly Daemon kill zone
    to_conv_w = to_conv_3d.reshape(8, 64)

    to_all_w = np.concatenate([to_raw_w, to_conv_w], axis=0)

    from_raw_w = np.zeros((16, 8, 8), dtype=np.float32)
    for p in [1, 4, 2, 3, 0]:
        from_raw_w[p] += 1.5
    from_raw_w[5] += 0.5

    for r in range(8):
        for c in range(8):
            if r in (0, 7) or c in (0, 7):
                for p in range(6):
                    from_raw_w[p, r, c] += 2.5  # Outer perimeter penalty

    from_conv_w = np.zeros((8, 8, 8), dtype=np.float32)
    from_conv_w[7] += 4.0
    from_all_w = np.concatenate([from_raw_w.reshape(16, 64), from_conv_w.reshape(8, 64)], axis=0)

    # 4. Construct ONNX Computation Graph
    obs_info = helper.make_tensor_value_info('observation', TensorProto.FLOAT, ['batch', 16, 8, 8])
    policy_info = helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 4096])
    value_info = helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch'])

    nodes = []
    initializers = []

    w_conv_tensor = helper.make_tensor('W_conv', TensorProto.FLOAT, [8, 16, 3, 3], conv_w.flatten().tolist())
    b_conv_tensor = helper.make_tensor('B_conv', TensorProto.FLOAT, [8], conv_b.tolist())
    initializers.extend([w_conv_tensor, b_conv_tensor])
    nodes.append(helper.make_node(
        'Conv',
        inputs=['observation', 'W_conv', 'B_conv'],
        outputs=['conv_out'],
        pads=[1, 1, 1, 1],
        kernel_shape=[3, 3]
    ))
    nodes.append(helper.make_node('Relu', ['conv_out'], ['conv_relu']))
    nodes.append(helper.make_node('Concat', ['observation', 'conv_relu'], ['features_all'], axis=1))

    # Value Head
    shape_val_flat = helper.make_tensor('shape_val_flat', TensorProto.INT64, [2], [-1, 1536])
    initializers.append(shape_val_flat)
    nodes.append(helper.make_node('Reshape', ['features_all', 'shape_val_flat'], ['val_flat']))

    w_val_tensor = helper.make_tensor('W_val', TensorProto.FLOAT, [1536, 1], val_w_total.flatten().tolist())
    initializers.append(w_val_tensor)
    nodes.append(helper.make_node('MatMul', ['val_flat', 'W_val'], ['val_raw']))
    nodes.append(helper.make_node('Tanh', ['val_raw'], ['val_tanh']))

    axes_1 = helper.make_tensor('axes_1', TensorProto.INT64, [1], [1])
    initializers.append(axes_1)
    nodes.append(helper.make_node('Squeeze', ['val_tanh', 'axes_1'], ['value']))

    # Policy Head
    shape_24_64 = helper.make_tensor('shape_24_64', TensorProto.INT64, [3], [-1, 24, 64])
    initializers.append(shape_24_64)
    nodes.append(helper.make_node('Reshape', ['features_all', 'shape_24_64'], ['feat_24_64']))

    channel_axis = helper.make_tensor('channel_axis', TensorProto.INT64, [1], [1])
    initializers.append(channel_axis)

    w_to_tensor = helper.make_tensor('W_to', TensorProto.FLOAT, [1, 24, 64], to_all_w.tolist())
    initializers.append(w_to_tensor)
    nodes.append(helper.make_node('Mul', ['feat_24_64', 'W_to'], ['to_elem']))
    nodes.append(helper.make_node('ReduceSum', ['to_elem', 'channel_axis'], ['to_score'], keepdims=1))

    w_from_tensor = helper.make_tensor('W_from', TensorProto.FLOAT, [1, 24, 64], from_all_w.tolist())
    initializers.append(w_from_tensor)
    nodes.append(helper.make_node('Mul', ['feat_24_64', 'W_from'], ['from_elem']))
    nodes.append(helper.make_node('ReduceSum', ['from_elem', 'channel_axis'], ['from_score_1d'], keepdims=1))
    nodes.append(helper.make_node('Transpose', ['from_score_1d'], ['from_score'], perm=[0, 2, 1]))

    nodes.append(helper.make_node('Add', ['from_score', 'to_score'], ['grid_scores']))
    shape_4096 = helper.make_tensor('shape_4096', TensorProto.INT64, [2], [-1, 4096])
    initializers.append(shape_4096)
    nodes.append(helper.make_node('Reshape', ['grid_scores', 'shape_4096'], ['policy']))

    graph = helper.make_graph(
        nodes=nodes,
        name='JenniferTacticalConvNet',
        inputs=[obs_info],
        outputs=[policy_info, value_info],
        initializer=initializers
    )

    if output_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_file = repo_root / "models" / "jennifer_tournament.onnx"
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
    print(f"Successfully generated Jennifer CNN model:")
    print(f"  Target:     {out_file}")
    print(f"  Size:       {size_kb:.2f} KB")
    print(f"  Opset / IR: Opset 18 (IR Version 9)")
    print(f"  Inputs:     observation [batch, 16, 8, 8]")
    print(f"  Outputs:    policy [batch, 4096], value [batch]")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and export the Jennifer Colony: One Tactical CNN ONNX model."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Custom destination path for the exported .onnx model."
    )
    args = parser.parse_args()
    build_jennifer_cnn(args.output)


if __name__ == '__main__':
    main()
