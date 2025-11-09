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

# Helper to read Hugging Face token from either HUGGINGFACE_TOKEN or HUGGINGFACE_TOKEN1
def get_hf_token():
    return os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACE_TOKEN1')
import tempfile
import subprocess
import shutil

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
        token = get_hf_token()
        if not token:
            raise RuntimeError('HUGGINGFACE_TOKEN not set')

        model_id = 'runwayml/stable-diffusion-v1-5'
        # If a custom endpoint is provided via HF_INFERENCE_ENDPOINT, try it first.
        hf_endpoint = os.getenv('HF_INFERENCE_ENDPOINT')
        if hf_endpoint:
            try:
                headers = {'Authorization': f'Bearer {token}'} if token else {}
                payload = {'inputs': prompt}
                r = requests.post(hf_endpoint, headers=headers, json=payload, timeout=120)
                if r.status_code == 200:
                    ct = r.headers.get('content-type', '')
                    if ct.startswith('image/'):
                        return Image.open(io.BytesIO(r.content)).convert('RGB')
                    try:
                        j = r.json()
                        if isinstance(j, dict) and 'data' in j and isinstance(j['data'], list) and len(j['data'])>0:
                            b64 = j['data'][0]
                            return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
                    except Exception:
                        pass
                # if not 200, capture and fall through to other attempts
                # record as an early error in the diagnostics list
                errors = [("HF_INFERENCE_ENDPOINT", f"status={getattr(r, 'status_code', None)}, text={getattr(r, 'text', '')[:300]}")]
            except Exception as e:
                errors = [("HF_INFERENCE_ENDPOINT_CALL", str(e))]

        # We'll try several approaches in order and surface detailed errors
        # so callers can see what failed. Methods tried:
        # 1) huggingface_hub.InferenceClient.text_to_image (higher-level)
        # 2) huggingface_hub.InferenceApi(...)(prompt, raw_response=True)
        # 3) direct POST to router.huggingface.co/hf-inference

        # preserve any early errors from custom endpoint attempt
        if 'errors' not in locals():
            errors = []

        # Attempt 1: InferenceClient (modern high-level client)
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=token)
            try:
                res = client.text_to_image(prompt, model=model_id)
                if isinstance(res, (bytes, bytearray)):
                    return Image.open(io.BytesIO(res)).convert('RGB')
                # Some clients return dict/list metadata — try to parse possible image bytes
                if isinstance(res, dict) and 'data' in res and isinstance(res['data'], list) and len(res['data'])>0:
                    b64 = res['data'][0]
                    try:
                        return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
                    except Exception:
                        pass
            except Exception as e:
                errors.append(('InferenceClient', str(e)))
        except Exception as e:
            errors.append(('InferenceClientImport', str(e)))

        # Attempt 2: InferenceApi with raw_response to inspect binary responses
        try:
            from huggingface_hub import InferenceApi
            api = InferenceApi(repo_id=model_id, token=token)
            try:
                resp = api(prompt, raw_response=True)
                status = getattr(resp, 'status_code', None)
                ct = resp.headers.get('content-type', '') if resp is not None else ''
                if status == 200 and ct.startswith('image/'):
                    return Image.open(io.BytesIO(resp.content)).convert('RGB')
                # If JSON with base64 payload
                try:
                    j = resp.json()
                    if isinstance(j, dict) and 'data' in j and isinstance(j['data'], list) and len(j['data'])>0:
                        b64 = j['data'][0]
                        return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
                except Exception:
                    pass
                errors.append(('InferenceApi', f'status={status}, content-type={ct}, text={resp.text[:300] if resp is not None else None}'))
            except Exception as e:
                errors.append(('InferenceApiCall', str(e)))
        except Exception as e:
            errors.append(('InferenceApiImport', str(e)))

        # Attempt 3: direct router POST
        try:
            router_url = 'https://router.huggingface.co/hf-inference'
            headers = {'Authorization': f'Bearer {token}'}
            alt_payload = {'model': model_id, 'inputs': prompt, 'options': {'wait_for_model': True}}
            r = requests.post(router_url, headers=headers, json=alt_payload, timeout=120)
            if r.status_code == 200:
                ct = r.headers.get('content-type', '')
                if ct.startswith('image/'):
                    return Image.open(io.BytesIO(r.content)).convert('RGB')
                try:
                    j = r.json()
                    if isinstance(j, dict) and 'data' in j and isinstance(j['data'], list) and len(j['data'])>0:
                        b64 = j['data'][0]
                        return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
                except Exception:
                    pass
            errors.append(('router', f'status={r.status_code}, text={r.text[:300]}'))
        except Exception as e:
            errors.append(('routerCall', str(e)))

        # As a last-ditch fallback, try the legacy api-inference endpoint (likely 410)
        try:
            url = f'https://api-inference.huggingface.co/models/{model_id}'
            headers = { 'Authorization': f'Bearer {token}' }
            payload = { 'inputs': prompt, 'options': { 'wait_for_model': True } }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            ct = resp.headers.get('content-type', '')
            if resp.status_code == 200 and ct.startswith('image/'):
                return Image.open(io.BytesIO(resp.content)).convert('RGB')
            try:
                j = resp.json()
                if isinstance(j, dict) and 'data' in j and isinstance(j['data'], list) and len(j['data'])>0:
                    b64 = j['data'][0]
                    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
            except Exception:
                pass
            errors.append(('legacy_api', f'status={resp.status_code}, text={resp.text[:300]}'))
        except Exception as e:
            errors.append(('legacyCall', str(e)))

        # All attempts failed — raise an error with collected diagnostics so callers can see why
        raise RuntimeError('Hugging Face generation failed; attempts: ' + json.dumps(errors[:10], default=str))

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
    api_available = bool(get_hf_token())
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
        hf_diagnostics = None
        # If a Hugging Face token is provided, prefer the hosted API for real images
        hf_token = get_hf_token()
        if hf_token:
            try:
                image = engine.generate_with_hf(prompt)
                source = 'huggingface'
            except Exception as e:
                # Capture diagnostics so the frontend can display why HF failed
                hf_diagnostics = str(e)
                print(f"❌ Hugging Face API error: {hf_diagnostics}")
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
            "hf_diagnostics": hf_diagnostics,
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


@app.route('/api/render-video', methods=['POST'])
def render_video():
    try:
        payload = request.json or {}
        images = payload.get('images', [])
        fps = int(payload.get('fps', 12))
        codec = payload.get('codec', 'libx264')

        if not images:
            return jsonify({'status': 'error', 'message': 'no images provided'}), 400

        # Ensure ffmpeg is available
        if shutil.which('ffmpeg') is None:
            return jsonify({'status': 'error', 'message': 'ffmpeg not found on server'}), 500

        # Create temporary working directory
        workdir = tempfile.mkdtemp(prefix='realityblur_video_')
        try:
            # Write frames as PNGs
            for i, dataurl in enumerate(images):
                # dataurl expected like 'data:image/png;base64,....'
                if ',' in dataurl:
                    _, b64 = dataurl.split(',', 1)
                else:
                    b64 = dataurl
                data = base64.b64decode(b64)
                fname = Path(workdir) / f'frame{i:04d}.png'
                with open(fname, 'wb') as f:
                    f.write(data)

            out_path = Path(workdir) / 'out.mp4'
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-framerate', str(fps),
                '-i', str(Path(workdir) / 'frame%04d.png'),
                '-c:v', codec,
                '-pix_fmt', 'yuv420p',
                str(out_path)
            ]
            subprocess.check_call(cmd)

            # Read output and return as base64 data URL
            with open(out_path, 'rb') as f:
                vbytes = f.read()
            v64 = base64.b64encode(vbytes).decode()
            return jsonify({'status': 'success', 'video': f'data:video/mp4;base64,{v64}'})
        finally:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass

    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'message': f'ffmpeg failed: {e}'}), 500
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
    # Allow overriding the port via the PORT environment variable so we can
    # avoid collisions during development (e.g., use PORT=5001).
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, host='0.0.0.0', port=port)
