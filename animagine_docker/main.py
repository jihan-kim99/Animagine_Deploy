import torch
from flask import Flask, request, jsonify
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from io import BytesIO
import base64

from huggingface_hub import login

app = Flask(__name__)
login("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")

print('logged in...')

model = "linaqruf/animagine-xl"

print("Loading model...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    )
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

print("Model loaded")

@app.route('/ping', methods=['GET'])
def ping():
  return jsonify({'message': 'pong'})

@app.route('/generate_image', methods=['POST'])
def generate_image():
  try:
    data = request.json
    prompt = data['prompt']
    negative_prompt = data['negative_prompt']
    image_data = generate_image_from_prompt(prompt, negative_prompt)
    buffered = BytesIO()
    image_data.save(buffered, format="JPEG")
    image_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'image': image_string})
  except Exception as e:
    return jsonify({'error': str(e)}), 500

def generate_image_from_prompt(prompt, negative_prompts):
  image = pipe(prompt, negative_prompt=negative_prompts, width=1024, height=1024, guidance_scale=7.5)
  return image

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)