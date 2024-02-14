from transformers import AutoTokenizer,PreTrainedTokenizerBase

t1 = AutoTokenizer.from_pretrained('hkunlp/instructor-large', model_max_length=512)

a = 'hello, world, will tomorrow be better?'
print(len(t1(a)['input_ids']))