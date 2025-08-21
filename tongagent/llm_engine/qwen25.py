from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list
from tongagent.utils import load_config
import torch, os
from typing import Optional
from PIL import Image

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor
)

# Qwen2.5-VL classes live under qwen2_5_vl in recent transformers
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # transformers ≥ 4.51
    _HAS_QWEN25_VL = True
except Exception:
    _HAS_QWEN25_VL = False

from qwen_vl_utils import process_vision_info
from openai import OpenAI


def load_pretrained_model(model_name: str):
    """Load Qwen 2.5 LLM or Qwen2.5-VL depending on model_name."""
    torch.manual_seed(0)
    print("from pretrained", model_name)

    if "VL" in model_name.upper():
        assert _HAS_QWEN25_VL, (
            "Qwen2.5-VL requires transformers >= 4.51.x. "
            "Please upgrade: pip install -U transformers accelerate"
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",  # optional; remove if FA2 not installed
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    # Qwen 2.5 LLM (e.g., Qwen/Qwen2.5-7B-Instruct)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_client_model(endpoint: str):
    # OpenAI-compatible client (for hosted/inference server path)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=f"http://{endpoint}:8000/v1",
    )
    return client


class ModelSingleton:
    def __new__(cls, model_name, lora_path=None):
        if hasattr(cls, "model_dict") and model_name in cls.model_dict:
            return cls
        if not hasattr(cls, "model_dict"):
            cls.model_dict = {}

        if "VL" in model_name.upper():
            model, processor = load_pretrained_model(model_name)
            if lora_path is not None:
                from peft.peft_model import PeftModel
                model = PeftModel.from_pretrained(model, lora_path)
                model.merge_and_unload()
            cls.model_dict[model_name] = (model, processor)
        else:
            config = load_config()
            model = load_client_model(config.qwen25.endpoint)
            tokenizer = None
            cls.model_dict[model_name] = (model, tokenizer)
        return cls


openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class Qwen25Engine(HfApiEngine):
    def __init__(self, model_name: str = "", lora_path: Optional[str] = None):
        module = ModelSingleton(model_name, lora_path)
        self.has_vision = "VL" in model_name.upper()
        self.model, self.tokenizer_or_processor = module.model_dict[model_name]
        self.model_name = model_name

        if self.has_vision:
            self.processor = self.tokenizer_or_processor  # processor for VL
        else:
            self.tokenizer = self.tokenizer_or_processor  # None for client path

    def call_llm(self, messages, stop_sequences=None, *args, **kwargs):
        stop_sequences = stop_sequences or []
        # Convert MessageRole → OpenAI dicts
        msgs = []
        for m in messages:
            if m["role"] == MessageRole.SYSTEM:
                msgs.append({"role": "system", "content": m["content"]})
            elif m["role"] == MessageRole.USER:
                msgs.append({"role": "user", "content": m["content"]})
            else:
                msgs.append({"role": "assistant", "content": m["content"]})

        response = self.model.chat.completions.create(
            model=self.model_name, messages=msgs, stop=stop_sequences
        )
        return response.choices[0].message.content

    def call_vlm(self, messages, stop_sequences=None, *args, **kwargs):
        stop_sequences = stop_sequences or []
        image_paths = kwargs.get("image_paths", [])

        # Inject images into the first user turn
        for i, m in enumerate(messages):
            if m["role"] == MessageRole.USER:
                content = []
                for p in image_paths:
                    content.append({"type": "image", "image": p})
                content.append({"type": "text", "text": m["content"]})
                messages[i] = {"role": "user", "content": content}
                break

        # Use chat template for Qwen2.5-VL
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=100,
            do_sample=True,
            repetition_penalty=1.05,
        )

        # Trim prompt tokens from outputs
        out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        out_text = self.processor.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Respect stop sequences
        for s in stop_sequences:
            j = out_text.find(s)
            if j != -1:
                out_text = out_text[:j]
                break
        return out_text

    def __call__(self, messages, stop_sequences=None, *args, **kwargs) -> str:
        stop_sequences = stop_sequences or []
        torch.cuda.empty_cache()

        # Normalize messages (transformers agents)
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        if self.has_vision:
            return self.call_vlm(messages, stop_sequences=stop_sequences, **kwargs)
        else:
            return self.call_llm(messages, stop_sequences=stop_sequences, **kwargs)


if __name__ == "__main__":
    # LLM path
    model, tok = load_pretrained_model("Qwen/Qwen2.5-7B-Instruct")
    print("Loaded:", type(model).__name__)
