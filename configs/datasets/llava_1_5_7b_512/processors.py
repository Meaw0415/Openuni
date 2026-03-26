from transformers import AutoTokenizer

llava_model_name_or_path = '/gemini/user/private/LLM-CKPT/llava-1.5-7b-hf'

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llava_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

prompt_template = dict(
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
    IMG_START_TOKEN='<img>',
    IMG_END_TOKEN='</img>',
    INSTRUCTION='USER: {input}',
    GENERATION='ASSISTANT: {output}',
    CFG='')

image_size = 512
pad_index = -100
