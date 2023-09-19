# loftr2onnx
Translate loftr indoor and outdoor weights to onnx model

# step1
Create a 'weights' folder and download the weights from the LOFTR project's link to your local machine.
[https://github.com/zju3dv/LoFTR]
weights link [https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf]

# step2
Configure the environment and run the file. 

    python convert_to_onnx.py  --model_path your_weight_path  #default='.\weights\outdoor_ds.ckpt'
also you can edit convert_to_onnx.py to change your onnx_model.name

# step3
file of test onnx_model inference time is coming soon....
