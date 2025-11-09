from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
from PIL import Image, ImageColor
import torch
import os
import requests
import time
from typing import Any, cast
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

class RealityBlurEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")
        # Annotate as Any so static type checkers don't try to validate
        # call signatures or attributes on the diffusers pipeline object.
        # Use a type comment to remain compatible with environments that
        # don't accept inline variable annotations in this position.
        self.pipe = None  # type: Any
        # Only load the heavy Stable Diffusion pipeline if a GPU is available.
        # Loading and running the full model on CPU in Codespaces or small machines
        # can cause native crashes or out-of-memory errors. We'll disable the
        # pipeline on CPU and use a safe fallback image instead.
        if self.device == "cuda":
            self.load_model()
        else:
            print("ℹ️ GPU not available — skipping heavy model load. Using fallback mode.")
    
    def load_model(self):
        """Load a simple AI model to start with"""
        try:
            from diffusers import StableDiffusionPipeline
            print("📦 Loading Stable Diffusion...")
            
            # Use a smaller model for faster loading
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            self.pipe = self.pipe.to(self.device)
            print("✅ AI model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.pipe = None

    def generate_with_hf(self, prompt: str):
        """Generate an image using the Hugging Face Inference API.

        Returns a PIL.Image on success or raises an Exception on failure.
        """
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            raise RuntimeError('HUGGINGFACE_TOKEN not set')

        model_id = 'runwayml/stable-diffusion-v1-5'
        # Try to use the official huggingface_hub client (InferenceApi) which
        # handles routing/endpoints; fall back to a raw HTTP request if the
        # client isn't available.
        try:
            from huggingface_hub import InferenceApi
            api = InferenceApi(repo_id=model_id, token=token)
            # The client returns bytes for image-producing models.
            result = api(inputs=prompt, options={"wait_for_model": True})

            # If result is bytes, open as image
            if isinstance(result, (bytes, bytearray)):
                return Image.open(io.BytesIO(result)).convert('RGB')

            # If result is a dict/list, attempt to parse similar to previous logic
            if isinstance(result, dict):
                if 'error' in result:
                    raise RuntimeError(f"Hugging Face API error: {result['error']}")
                if 'data' in result and isinstance(result['data'], list) and len(result['data'])>0:
                    b64 = result['data'][0]
                    try:
                        return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
                    except Exception:
                        pass
            if isinstance(result, list) and len(result)>0 and isinstance(result[0], (bytes, bytearray)):
                return Image.open(io.BytesIO(result[0])).convert('RGB')

        except Exception:
            # If the client import or call fails, fall back to raw HTTP route
            pass

        # Fallback: raw HTTP call (legacy behavior). This will attempt the
        # classical api-inference endpoint, and if it returns 410, try router.
        url = f'https://api-inference.huggingface.co/models/{model_id}'
        headers = { 'Authorization': f'Bearer {token}' }
        payload = { 'inputs': prompt, 'options': { 'wait_for_model': True } }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            if resp.status_code == 410:
                try:
                    router_url = 'https://router.huggingface.co/hf-inference'
                    alt_payload = { 'model': model_id, 'inputs': prompt, 'options': { 'wait_for_model': True } }
                    resp2 = requests.post(router_url, headers=headers, json=alt_payload, timeout=120)
                    resp = resp2
                except Exception:
                    pass

            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f'Hugging Face API error: {resp.status_code} {err}')

        content_type = resp.headers.get('content-type', '')
        if content_type.startswith('image/'):
            return Image.open(io.BytesIO(resp.content)).convert('RGB')

        try:
            data = resp.json()
            if isinstance(data, dict) and 'error' in data:
                raise RuntimeError(f"Hugging Face API error: {data['error']}")
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list) and len(data['data'])>0:
                b64 = data['data'][0]
                try:
                    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
                except Exception:
                    pass
        except Exception:
            pass

        raise RuntimeError('Unexpected response from Hugging Face Inference API')

    def generate_image(self, prompt):
        """Generate image from text prompt"""
        if self.pipe is None:
            # Create a simple fallback image (placeholder) when pipeline isn't available
            from PIL import ImageDraw, ImageFont
            # Use ImageColor.getrgb so type checkers don't complain about
            # string color values; this produces an (R,G,B) tuple.
            img = Image.new('RGB', (512, 512), color=cast(Any, ImageColor.getrgb('#0f0f0f')))
            draw = ImageDraw.Draw(img)
            try:
                # Use a basic system font if available
                font = ImageFont.load_default()
                draw.text((16, 240), "Fallback image (model disabled)", fill=cast(Any, ImageColor.getrgb('#00ff88')), font=font)
            except Exception:
                pass
            return img
        
        try:
            with torch.no_grad():
                # Cast to Any to avoid static type-checker errors about
                # DiffusionPipeline call signatures and result shapes.
                pipe_any = cast(Any, self.pipe)
                result = pipe_any(
                    prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20,  # Faster generation
                    guidance_scale=7.5
                )
            # Access images dynamically; at runtime the pipeline result
            # exposes an `images` attribute.
            return result.images[0]
        except Exception as e:
            print(f"❌ Generation error: {e}")
            # Fallback
            return Image.new('RGB', (512, 512), color=cast(Any, ImageColor.getrgb('#ff4444')))

@app.route('/')
def home():
    return jsonify({
        "message": "🎭 RealityBlur AI Server is running!",
        "status": "healthy",
        "device": engine.device
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    # Indicate whether a hosted API token is configured
    api_available = bool(os.getenv('HUGGINGFACE_TOKEN'))
    return jsonify({
        "status": "healthy",
        "device": engine.device,
        "model_loaded": engine.pipe is not None,
        "api_available": api_available
    })

@app.route('/api/generate', methods=['POST'])
def generate_video():
    try:
        # request.json may be None if the client sent no JSON — guard against it.
        data = request.json or {}
        prompt = data.get('prompt', 'A realistic scene')
        
        print(f"🎬 Generating for prompt: {prompt}")
        start = time.time()
        source = 'fallback'
        # If a Hugging Face token is provided, prefer the hosted API for real images
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            try:
                image = engine.generate_with_hf(prompt)
                source = 'huggingface'
            except Exception as e:
                print(f"❌ Hugging Face API error: {e}")
                # fallback to local engine (which may be placeholder)
                image = engine.generate_image(prompt)
                # if local pipeline exists, mark as local; otherwise keep fallback
                source = 'fallback' if engine.pipe is None else 'local'
        else:
            # Generate image locally (fallback or real model if loaded)
            image = engine.generate_image(prompt)
            source = 'fallback' if engine.pipe is None else 'local'

        duration_ms = int((time.time() - start) * 1000)
        
        # Convert to base64 for frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "status": "success",
            "source": source,
            "generation_time_ms": duration_ms,
            "image": f"data:image/png;base64,{img_str}",
            "message": "🎭 Phase 1: Image generation working! Video coming next..."
        })
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Simple persistence for flows: save/load JSON files under backend/flows
FLOWS_DIR = Path(__file__).parent / 'flows'
FLOWS_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/api/flows', methods=['POST'])
def save_flow():
    try:
        payload = request.json or {}
        name = payload.get('name') or f'flow_{int(time.time())}'
        nodes = payload.get('nodes', [])
        edges = payload.get('edges', [])
        fname = f"{int(time.time())}_{name}.json"
        p = FLOWS_DIR / fname
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'name': name, 'nodes': nodes, 'edges': edges}, f, indent=2)
        return jsonify({'status': 'ok', 'name': fname})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/flows', methods=['GET'])
def list_flows():
    try:
        items = [p.name for p in sorted(FLOWS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True) if p.is_file()]
        return jsonify(items)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/flows/<flow_name>', methods=['GET'])
def load_flow(flow_name):
    try:
        p = FLOWS_DIR / flow_name
        if not p.exists():
            return jsonify({'status': 'error', 'message': 'not found'}), 404
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Initialize the engine
engine = RealityBlurEngine()

if __name__ == '__main__':
    print("🚀 Starting RealityBlur AI Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
