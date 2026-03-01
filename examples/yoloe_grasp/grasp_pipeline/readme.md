CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/grasp_pipeline.py --visualize 


可能是相机深度不好，导致 graspnet 识别的 grasp 不准。可以换lingbot-depth模型试试