# CUDA Diagnostics

Overall: PASS

- **cuda_available**: PASS — torch.cuda.is_available()=True, device_count=1
- **device_name_memory**: PASS — name=NVIDIA GeForce RTX 3050 6GB Laptop GPU, capability=(8,6), total_mem=6.44GB
- **tensor_op_cuda**: PASS — matmul_2048 mean=12.844 ms, out=(2048, 2048)
- **model_load_cuda**: PASS — YOLOv8s moved to cuda:0
- **inference_pass_cuda**: PASS — inference on 640x640 blank took 2232.5 ms, 0 detections (blank frame)

JSON: `D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection\helmet_anpr_research_rebuilt\reports\cuda_diagnostics.json`
