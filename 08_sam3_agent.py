import torch
import os
from functools import partial
from IPython.display import display, Image
from sam3.sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.sam3.agent.inference import run_single_image_inference
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# 参数配置
sam3_root = "./sam3"
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(
    checkpoint_path="./models/facebook/sam3/sam3.pt",
    bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)
LLM_CONFIGS = {
    "qwen3_vl_2b_thinking": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-2B-Thinking",
    }, 
}
model = "qwen3_vl_2b_thinking"
llm_config = LLM_CONFIGS[model]
llm_config["api_key"] = "DUMMY_API_KEY"
llm_config["name"] = model
LLM_SERVER_URL = "http://127.0.0.1:8000/v1"
# 测试单张图片推理
image = f"{sam3_root}/assets/images/test_image.jpg"
prompt = "the leftmost child wearing blue vest"
image = os.path.abspath(image)
send_generate_request = partial(
    send_generate_request_orig, 
    server_url=LLM_SERVER_URL, 
    model=llm_config["model"], 
    api_key=llm_config["api_key"])
call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
output_image_path = run_single_image_inference(
    image, prompt, llm_config, send_generate_request, call_sam_service, 
    debug=True, output_dir="agent_output"
)
# 显示结果图片
if output_image_path is not None:
    display(Image(filename=output_image_path))