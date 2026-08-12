from huggingface_hub import HfApi
api = HfApi()
datasets = api.list_datasets(search="asl", limit=10)
for d in datasets:
    print(d.id.encode('utf-8'))
