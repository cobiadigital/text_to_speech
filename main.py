import torch
from transformers import pipeline

torch.cuda.is_available()

url='https://app.callrail.com/calls/CALff850983ccdc441d8c0d7e1aa515cb42/recording/redirect?access_key=e093a54f9424c9e61394'
pytorch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current PyTorch device is set to", pytorch_device)


pipe = pipeline(model="openai/whisper-small", device=pytorch_device)
predictions = pipe(url, chunk_length_s=20, stride_length_s=(5))

predictions = predictions["text"]
print(predictions)