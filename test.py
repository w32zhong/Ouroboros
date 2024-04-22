import torch
from ouroboros import ouroboros
from transformers import AutoTokenizer
from ouroboros.models import LlamaForCausalLM
import time

window_size = 20
guess_set_size = 20
lookahead_level = 7
gamma = 12

drafter_path = "JackFram/llama-68m"
drafter_path = "JackFram/llama-160m"

small_model = LlamaForCausalLM.from_pretrained(drafter_path, torch_dtype=torch.float16, device_map='cuda', load_in_8bit=True)
target_model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map='cuda', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

prompt = "[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]"

input_ids = tokenizer(prompt, return_tensors='pt').to('cuda')['input_ids']

start_time = time.time()
ouroboros_output = ouroboros(input_ids, small_model, target_model, tokenizer, max_len=1900, gamma=gamma, window_size=window_size, guess_set_size=guess_set_size, lookahead_level=lookahead_level)
time_ouroboros = time.time() - start_time
output_tokens = ouroboros_output[0, input_ids.shape[-1]:]
cnt_tokens = len(output_tokens)
#print(tokenizer.decode(output_tokens))
print('ouroboros e2e speed:', time_ouroboros, cnt_tokens, cnt_tokens / time_ouroboros)

start_time = time.time()
std_output = target_model.generate(input_ids, do_sample=False, max_length=1900)
time_base = time.time() - start_time

print('same output?', ouroboros_output[:,:64].equal(std_output[:,:64]))
print('base e2e speed:', time_base, cnt_tokens, cnt_tokens / time_base)
