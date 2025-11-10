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

# Preferred model priority list — try these in order until one succeeds
MODEL_PRIORITY_LIST = [
    'black-forest-labs/FLUX.1-schnell',
    'runwayml/stable-diffusion-v1-5',
    'stabilityai/stable-diffusion-2-1',
    'CompVis/stable-diffusion-v1-4'
]

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

    def generate_with_hf(self, prompt: str, model_id: str = 'runwayml/stable-diffusion-v1-5'):
        """Generate an image using the Hugging Face Inference API.

        Returns a PIL.Image on success or raises an Exception on failure.
        """
        token = get_hf_token()
        if not token:
            raise RuntimeError('HUGGINGFACE_TOKEN not set')

    # model_id is provided by caller (default above)
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
                # Some clients may return a PIL Image object directly
                try:
                    from PIL import Image as PILImage
                    if isinstance(res, PILImage.Image):
                        return res.convert('RGB')
                except Exception:
                    pass
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

    def generate_with_smart_fallback(self, prompt: str):
        """Try a prioritized list of models for hosted inference and return the first successful image and model id.

        Returns (PIL.Image, model_id) on success or raises RuntimeError if none succeed.
        """
        token = get_hf_token()
        if not token:
            raise RuntimeError('HUGGINGFACE_TOKEN not set')

        errors = []
        for model in MODEL_PRIORITY_LIST:
            try:
                img = self.generate_with_hf(prompt, model_id=model)
                return img, model
            except Exception as e:
                errors.append((model, str(e)))
                # try next model
                continue

        # All models failed — raise with diagnostics
        raise RuntimeError('All HF models failed; attempts: ' + json.dumps(errors[:10], default=str))

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
        model_used = None
        # If a Hugging Face token is provided, prefer the hosted API for real images
        hf_token = get_hf_token()
        if hf_token:
            try:
                image, model_used = engine.generate_with_smart_fallback(prompt)
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
        
        resp = {
            "status": "success",
            "source": source,
            "hf_diagnostics": hf_diagnostics,
            "generation_time_ms": duration_ms,
            "image": f"data:image/png;base64,{img_str}",
            "message": "🎭 Phase 1: Image generation working! Video coming next..."
        }
        if model_used:
            resp['model_used'] = model_used
        return jsonify(resp)
        
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


def generate_varied_images(prompt: str, num_variations: int = 4):
    """Generate multiple varied images for the same prompt.
    Returns list of PIL.Image objects.
    """
    images = []
    hf_diagnostics = []
    models_used = []
    token = get_hf_token()
    for i in range(num_variations):
        varied_prompt = f"{prompt} - variation {i+1}"
        try:
            if token:
                try:
                    img, model = engine.generate_with_smart_fallback(varied_prompt)
                    models_used.append(model)
                except Exception as e:
                    hf_diagnostics.append(str(e))
                    img = engine.generate_image(varied_prompt)
            else:
                img = engine.generate_image(varied_prompt)
        except Exception as e:
            # On unexpected error, produce a fallback image instead of failing entirely
            print(f"❌ Error generating variation {i}: {e}")
            img = Image.new('RGB', (512, 512), color=cast(Any, ImageColor.getrgb('#444444')))
        images.append(img)
    return images, (hf_diagnostics or None), models_used


@app.route('/api/generate-variations', methods=['POST'])
def api_generate_variations():
    try:
        data = request.json or {}
        prompt = data.get('prompt', 'A realistic scene')
        num = int(data.get('num_variations', 4))
        num = max(1, min(12, num))
        imgs, hf_diag, models_used = generate_varied_images(prompt, num)

        out = []
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            out.append('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())

        return jsonify({'status': 'success', 'images': out, 'hf_diagnostics': hf_diag, 'models_used': models_used})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/render-dynamic-video', methods=['POST'])
def render_dynamic_video():
    """Render a dynamic video from multiple images applying short crossfade transitions.
    Expects JSON { images: [dataurl,...], fps: int, duration: float }
    """
    try:
        payload = request.json or {}
        images = payload.get('images', [])
        fps = int(payload.get('fps', 12))
        duration = float(payload.get('duration', max(3, len(images))))
        effects = payload.get('effects', []) or []
        transitions = payload.get('transitions', []) or []
        preset = (payload.get('preset') or payload.get('profile'))
        motion_blur = bool(payload.get('motion_blur', False))
        film_grain = float(payload.get('film_grain', 0.0) or 0.0)
        color_grade = payload.get('color_grade', None)
        sharpening = float(payload.get('sharpening', 0.0) or 0.0)
        aspect = payload.get('aspect_ratio') or payload.get('aspect') or payload.get('aspectRatio')

        # Apply presets
        PRESETS = {
            "tiktok": {"fps": 30, "duration": 15, "aspect": "9:16"},
            "instagram": {"fps": 30, "duration": 30, "aspect": "1:1"},
            "youtube": {"fps": 30, "duration": 60, "aspect": "16:9"}
        }
        if preset and preset in PRESETS:
            p = PRESETS[preset]
            fps = int(p.get('fps', fps))
            duration = float(p.get('duration', duration))
            aspect = aspect or p.get('aspect')

        if not images:
            return jsonify({'status': 'error', 'message': 'no images provided'}), 400

        if shutil.which('ffmpeg') is None:
            return jsonify({'status': 'error', 'message': 'ffmpeg not found on server'}), 500

        workdir = tempfile.mkdtemp(prefix='realityblur_dyn_')
        try:
            # write inputs
            in_files = []
            for i, dataurl in enumerate(images):
                if ',' in dataurl:
                    _, b64 = dataurl.split(',', 1)
                else:
                    b64 = dataurl
                data = base64.b64decode(b64)
                fname = Path(workdir) / f'img{i:03d}.png'
                with open(fname, 'wb') as f:
                    f.write(data)
                in_files.append(str(fname))

            n = len(in_files)

            # Compute per-image durations (allow variable timing via payload.durations)
            given_durations = payload.get('durations') or payload.get('timings') or []
            if given_durations and len(given_durations) >= n:
                segs = [max(0.5, float(x)) for x in given_durations[:n]]
            else:
                # Distribute duration unevenly for rhythm if requested
                if payload.get('variable_timing'):
                    # simple pattern: favor middle frames
                    segs = []
                    base = max(1.0, duration / n)
                    for i in range(n):
                        factor = 1.0 + (0.3 * (0.5 - abs((i - (n-1)/2)/((n-1)/2 if n>1 else 1))))
                        segs.append(max(0.6, base * factor))
                else:
                    segs = [max(1.0, duration / n) for _ in range(n)]

            # helper to create a short video for each image with optional ken burns
            segment_files = []
            out_w, out_h = None, None
            # derive output resolution from aspect and base width 720 for vertical social formats
            if aspect:
                w,h = (720, 1280) if aspect in ('9:16','9:16','vertical') else ((720,720) if aspect in ('1:1','square') else (1280,720))
                out_w, out_h = w,h
            else:
                out_w, out_h = 1280, 720

            for i, infile in enumerate(in_files):
                seg = float(segs[i])
                eff = effects[i] if i < len(effects) else {}
                # Ken Burns parameters
                if eff and eff.get('type') == 'ken_burns':
                    start_scale = float(eff.get('start_scale', 1.0))
                    end_scale = float(eff.get('zoom', eff.get('end_scale', 1.08)))
                    pan_x = float(eff.get('pan_x', eff.get('movement_x', 0.0)))
                    pan_y = float(eff.get('pan_y', eff.get('movement_y', 0.0)))
                else:
                    # default subtle ken burns
                    start_scale = 1.02
                    end_scale = 1.12
                    pan_x = 0.0
                    pan_y = 0.0

                frames = int(max(1, seg * fps))
                # zoompan expression: linear interpolation per frame
                # z = start + (end-start)*(on/(n-1))
                if frames>1:
                    z_expr = f"{start_scale} + ({end_scale - start_scale})*(on/{max(frames-1,1)})"
                else:
                    z_expr = f"{start_scale}"

                # compute pan expression in pixels: pan range based on movement fractions
                # x = (iw - iw/zoom)/2 + pan_x*(iw/zoom)
                x_expr = f"(iw - iw/({z_expr}))/2 + ({pan_x})*(iw/({z_expr}))"
                y_expr = f"(ih - ih/({z_expr}))/2 + ({pan_y})*(ih/({z_expr}))"

                seg_out = Path(workdir) / f'seg{i:03d}.mp4'
                vf = f"scale={out_w}:{out_h},zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={out_w}x{out_h},fps={fps},format=yuv420p"
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-loop', '1', '-t', str(seg), '-i', infile,
                    '-vf', vf,
                    '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', str(seg_out)
                ]
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as e:
                    # fallback: simple looped image to video
                    cmd2 = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-loop', '1', '-t', str(seg), '-i', infile, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(seg_out)]
                    subprocess.check_call(cmd2)
                segment_files.append(str(seg_out))

            # Now build the filter_complex to chain segments with transitions
            # We'll use xfade for available transition types and fallback to fade
            # Prepare inputs
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
            for sf in segment_files:
                cmd += ['-i', sf]

            filters = []
            last_label = '[0:v]'
            current_label = '0'
            # transition duration (choose small for social rhythm)
            for i in range(1, len(segment_files)):
                t_type = transitions[i-1] if i-1 < len(transitions) else 'fade'
                # map user-friendly names to xfade transitions (fallback to fade)
                mapping = {
                    'fade': 'fade', 'fade_black': 'fade', 'slide_left': 'slideleft', 'slide_right': 'slideright',
                    'wipe_up': 'wipeup', 'wipe_down': 'wipedown', 'zoom_rotate': 'slideleft'
                }
                xfade_name = mapping.get(t_type, 'fade')
                trans_dur = min(0.8, float(payload.get('transition_duration', 0.6)))
                offset = 0
                # compute offset as cumulative durations so far minus trans_dur
                offset = sum([float(segs[j]) for j in range(i)]) - trans_dur
                filt = f"{last_label}[{i}:v]xfade=transition={xfade_name}:duration={trans_dur}:offset={offset}[v{i}]"
                filters.append(filt)
                last_label = f"[v{i}]"

            filter_complex = ';'.join(filters)
            out_path = Path(workdir) / 'out.mp4'
            if not filter_complex:
                # single segment -> move to out
                shutil.copyfile(segment_files[0], out_path)
            else:
                full_cmd = cmd + ['-filter_complex', filter_complex, '-map', last_label, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out_path)]
                subprocess.check_call(full_cmd)

            # post processing: motion blur, film grain, color grade, sharpening
            post_in = str(out_path)
            post_tmp = Path(workdir) / 'out_post.mp4'
            post_filters = []
            if motion_blur:
                # simple temporal blend to emulate motion blur
                post_filters.append('tblend=all_mode=average')
            if film_grain and film_grain > 0.0:
                # noise: amount based on film_grain (0-0.5)
                mag = max(0.01, min(0.5, film_grain))
                # alls param controls strength; use integer like 12*mag
                alls = int(20 * mag)
                post_filters.append(f'noise=alls={alls}:allf=t')
            if color_grade == 'cinematic':
                # mild lift/gamma/sat adjustment
                post_filters.append('eq=contrast=1.08:brightness=0.01:saturation=1.06')
            if sharpening and sharpening > 0.0:
                post_filters.append(f'unsharp=5:5:{sharpening}')

            if post_filters:
                vf = ','.join(post_filters)
                cmd_post = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', post_in, '-vf', vf, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(post_tmp)]
                subprocess.check_call(cmd_post)
                final_path = post_tmp
            else:
                final_path = out_path

            with open(final_path, 'rb') as f:
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
