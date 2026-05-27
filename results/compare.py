import pandas as pd

baseline = pd.read_json(r'results/result_baseline.jsonl', lines=True)
# baseline.info()
print('baseline: format_reward, answer_reward, reward')
print(baseline['format_reward'].sum()/5000)
print(baseline['answer_reward'].sum()/5000)
print(baseline['reward'].sum()/5000)
print('------------------------------------------------')

sft = pd.read_json(r'results/result_sft.jsonl', lines=True)
# sft.info()
print('sft: format_reward, answer_reward, reward')
print(sft['format_reward'].sum()/5000)
print(sft['answer_reward'].sum()/5000)
print(sft['reward'].sum()/5000)