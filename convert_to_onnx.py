import os
import onnxruntime
import torch
from loftr_wrapper import LoFTRWrapper as LoFTR


def get_args():
    import argparse

    parser = argparse.ArgumentParser("convert loftr torch weights to onnx format")
    parser.add_argument("--model_path", type=str, default='.\weights\outdoor_ds.ckpt')

    return parser.parse_args()


def main():

    args = get_args()
    model_path = args.model_path
    model = LoFTR()
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()

    batch_size = 1
    height = 512
    width = 512

    data = {}
    data["image0"] = torch.randn(batch_size, 1, height, width)
    data["image1"] = torch.randn(batch_size, 1, height, width)

    torch.onnx.export(
        model,
        data,
        "loftr_outdoor_ds.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=list(data.keys()),
        output_names=["keypoints0", "keypoints1", "confidence"],
        dynamic_axes={
            "image0": {2: "height", 3: "width"},
            "image1": {2: "height", 3: "width"},
            "keypoints0": {0: "num_keypoints"},
            "keypoints1": {0: "num_keypoints"},
            "confidence": {0: "num_keypoints"},
        },
    )

    print(f"\nonnx model is saved to: {os.getcwd()}/loftr_outdoor_ds.onnx")

    print("\ntest inference using onnxruntime")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        onnx_providers = ['CUDAExecutionProvider']
    else:
        onnx_providers = ['CPUExecutionProvider']
    sess = onnxruntime.InferenceSession("loftr_outdoor_ds.onnx", providers=onnx_providers)
    for input in sess.get_inputs():
        print("input: ", input)

    print("\n")
    for output in sess.get_outputs():
        print("output: ", output)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
