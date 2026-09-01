"""
Model Helper & Standalone ONNX Exporter for Voice Spoof Detection.
Generates an ONNX model (`spoof_detector.onnx`) tailored for 2-second 16kHz audio classification.
Input: `input_audio` with shape [1, 32000] (Float32).
Output: `spoof_prob` with shape [1, 1] (Float32, Sigmoid activation in [0.0, 1.0]).
"""

import os
import sys
import time
import argparse
import numpy as np

DEFAULT_MODEL_PATH = "spoof_detector.onnx"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 32,000 samples


def create_onnx_model(output_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Constructs a valid, fully typed single-precision (TensorProto.FLOAT / np.float32)
    ONNX voice spoof detection model directly using `onnx.helper`.

    Graph Architecture:
      Input: 'input_audio' [1, 32000] (Float32)
      -> Reshape to [1, 1, 32000] for Conv1D
      -> Conv1D (in_channels=1, out_channels=16, kernel_size=64, stride=16)
      -> Relu
      -> Conv1D (in_channels=16, out_channels=32, kernel_size=32, stride=8)
      -> Relu
      -> GlobalAveragePool -> [1, 32, 1]
      -> Reshape / Flatten -> [1, 32]
      -> Gemm (Dense: 32 -> 1) -> [1, 1] (Logits)
      -> Sigmoid -> 'spoof_prob' [1, 1] (Float32 in [0.0, 1.0])
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        raise ImportError(
            "`onnx` package is required to construct the model. Run `pip install onnx`."
        )

    print(f"[*] Building single-precision Float32 ONNX graph with onnx.helper...")

    # Set seed for reproducible initializers
    np.random.seed(42)

    # 1. Inputs & Outputs Definitions (explicitly TensorProto.FLOAT)
    input_audio = helper.make_tensor_value_info(
        "input_audio", TensorProto.FLOAT, [1, WINDOW_SAMPLES]
    )
    spoof_prob = helper.make_tensor_value_info(
        "spoof_prob", TensorProto.FLOAT, [1, 1]
    )

    # 2. Initializers (Weights, Biases & Reshape constants explicitly np.float32 / np.int64)
    # Shape tensor for Reshape to [1, 1, 32000]
    reshape_shape = numpy_helper.from_array(
        np.array([1, 1, WINDOW_SAMPLES], dtype=np.int64), name="reshape_shape"
    )

    # Conv1 weights: 16 filters, 1 input channel, kernel size 64
    w_conv1 = (np.random.randn(16, 1, 64) * np.sqrt(2.0 / 64)).astype(np.float32)
    b_conv1 = np.zeros(16, dtype=np.float32)
    t_w_conv1 = numpy_helper.from_array(w_conv1, name="w_conv1")
    t_b_conv1 = numpy_helper.from_array(b_conv1, name="b_conv1")

    # Conv2 weights: 32 filters, 16 input channels, kernel size 32
    w_conv2 = (np.random.randn(32, 16, 32) * np.sqrt(2.0 / (16 * 32))).astype(np.float32)
    b_conv2 = np.zeros(32, dtype=np.float32)
    t_w_conv2 = numpy_helper.from_array(w_conv2, name="w_conv2")
    t_b_conv2 = numpy_helper.from_array(b_conv2, name="b_conv2")

    # Shape tensor for Flatten / Reshape to [1, 32]
    flat_shape = numpy_helper.from_array(
        np.array([1, 32], dtype=np.int64), name="flat_shape"
    )

    # Gemm (Dense Linear) weights: 32 input features -> 1 output logit
    w_fc = (np.random.randn(32, 1) * 0.1).astype(np.float32)
    b_fc = np.array([-0.5], dtype=np.float32)
    t_w_fc = numpy_helper.from_array(w_fc, name="w_fc")
    t_b_fc = numpy_helper.from_array(b_fc, name="b_fc")

    # 3. Computation Graph Nodes
    # Node 1: Reshape [1, 32000] -> [1, 1, 32000]
    node_reshape = helper.make_node(
        "Reshape",
        inputs=["input_audio", "reshape_shape"],
        outputs=["reshaped_audio"],
        name="node_reshape",
    )

    # Node 2: Conv1D (1 -> 16 channels, kernel=64, stride=16)
    node_conv1 = helper.make_node(
        "Conv",
        inputs=["reshaped_audio", "w_conv1", "b_conv1"],
        outputs=["conv1_out"],
        kernel_shape=[64],
        strides=[16],
        name="node_conv1",
    )

    # Node 3: ReLU 1
    node_relu1 = helper.make_node(
        "Relu",
        inputs=["conv1_out"],
        outputs=["relu1_out"],
        name="node_relu1",
    )

    # Node 4: Conv1D (16 -> 32 channels, kernel=32, stride=8)
    node_conv2 = helper.make_node(
        "Conv",
        inputs=["relu1_out", "w_conv2", "b_conv2"],
        outputs=["conv2_out"],
        kernel_shape=[32],
        strides=[8],
        name="node_conv2",
    )

    # Node 5: ReLU 2
    node_relu2 = helper.make_node(
        "Relu",
        inputs=["conv2_out"],
        outputs=["relu2_out"],
        name="node_relu2",
    )

    # Node 6: GlobalAveragePool -> [1, 32, 1]
    node_gap = helper.make_node(
        "GlobalAveragePool",
        inputs=["relu2_out"],
        outputs=["gap_out"],
        name="node_gap",
    )

    # Node 7: Reshape to [1, 32]
    node_flatten = helper.make_node(
        "Reshape",
        inputs=["gap_out", "flat_shape"],
        outputs=["flatten_out"],
        name="node_flatten",
    )

    # Node 8: Gemm (Dense: [1, 32] x [32, 1] + [1] -> [1, 1])
    node_gemm = helper.make_node(
        "Gemm",
        inputs=["flatten_out", "w_fc", "b_fc"],
        outputs=["logits"],
        alpha=1.0,
        beta=1.0,
        name="node_gemm",
    )

    # Node 9: Sigmoid activation -> [1, 1] (spoof probability constrained between 0.0 and 1.0)
    node_sigmoid = helper.make_node(
        "Sigmoid",
        inputs=["logits"],
        outputs=["spoof_prob"],
        name="node_sigmoid",
    )

    # 4. Assemble Graph and Model Proto
    graph = helper.make_graph(
        nodes=[
            node_reshape,
            node_conv1,
            node_relu1,
            node_conv2,
            node_relu2,
            node_gap,
            node_flatten,
            node_gemm,
            node_sigmoid,
        ],
        name="VoiceSpoofDetector",
        inputs=[input_audio],
        outputs=[spoof_prob],
        initializer=[
            reshape_shape,
            t_w_conv1,
            t_b_conv1,
            t_w_conv2,
            t_b_conv2,
            flat_shape,
            t_w_fc,
            t_b_fc,
        ],
    )

    # Create ONNX model with standard opset 14
    model = helper.make_model(
        graph,
        producer_name="VoiceSpoofDetector",
        opset_imports=[helper.make_opsetid("", 14)],
    )

    # Validate graph structure and type consistency
    onnx.checker.check_model(model)

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    print(f"[OK] Validated single-precision ONNX model successfully saved to: {output_path}")
    return output_path


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Checks if the ONNX model exists at `model_path`.
    If not, creates and validates it automatically.
    """
    if os.path.exists(model_path):
        return model_path

    print(f"[!] Model not found at '{model_path}'. Generating model graph...")
    return create_onnx_model(model_path)


def test_model_inference(model_path: str = DEFAULT_MODEL_PATH):
    """
    Runs a test inference and benchmark using ONNX Runtime.
    """
    import onnxruntime as ort

    print(f"\n--- Testing ONNX Model Inference on '{model_path}' ---")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"Model Input  : '{input_info.name}' (Shape: {input_info.shape}, Type: {input_info.type})")
    print(f"Model Output : '{output_info.name}' (Shape: {output_info.shape}, Type: {output_info.type})")

    # Generate 2.0s synthetic 16kHz audio input (1, 32000) float32
    t = np.linspace(0, WINDOW_SECONDS, WINDOW_SAMPLES, endpoint=False, dtype=np.float32)
    synthetic_audio = (0.6 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(WINDOW_SAMPLES)).astype(np.float32)
    input_tensor = np.expand_dims(synthetic_audio, axis=0)  # Shape: (1, 32000), dtype: float32

    # Warmup
    _ = session.run([output_info.name], {input_info.name: input_tensor})

    # Benchmark 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        outputs = session.run([output_info.name], {input_info.name: input_tensor})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    spoof_prob = float(outputs[0][0][0])
    bonafide_prob = 1.0 - spoof_prob

    print(f"Spoof Confidence    : {spoof_prob:.4f} ({spoof_prob * 100:.1f}%)")
    print(f"Bonafide Confidence : {bonafide_prob:.4f} ({bonafide_prob * 100:.1f}%)")
    print(f"Decision Label      : {'SPOOF' if spoof_prob >= 0.5 else 'BONAFIDE'}")
    print(f"Average Latency     : {avg_lat:.2f} ms (Min: {min_lat:.2f} ms)")
    print(f"Sub-50ms Target Met : {'YES [PASS]' if avg_lat < 50.0 else 'NO [FAIL]'}")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Spoof ONNX Model Generator & Benchmark")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help="Path for output ONNX file")
    parser.add_argument("--test", action="store_true", help="Run inference benchmark on generated model")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing model")
    args = parser.parse_args()

    if args.force and os.path.exists(args.output):
        os.remove(args.output)
        print(f"[*] Deleted existing model: {args.output}")

    model_file = ensure_model_exists(args.output)
    try:
        test_model_inference(model_file)
    except Exception as e:
        print(f"[!] Warning: Could not run test inference: {e}")
