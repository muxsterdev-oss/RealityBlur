# 🎭 RealityBlur AI

Blurring the line between AI-generated content and reality.

## 🚀 Quick Start in GitHub Codespaces

1. **Open in Codespaces** - Click "Code" → "Codespaces" → "Create codespace on main"
2. **Wait for setup** - Dependencies install automatically
3. **Start development** - Run: `bash start-development.sh`
4. **Open ports** - Check the "Ports" tab and open ports 3000 (frontend) and 5000 (backend)

## 🎯 Current Features

- ✅ Node-based UI foundation
- ✅ Text-to-image generation
- ✅ Real-time preview
- ✅ Cloud-based development

## 🔮 Coming Next

- 🎬 Real video generation
- 🔍 Advanced realism enhancements
- ⚡ Temporal consistency
- 🎨 Node-based workflow editor

## 🔐 Hugging Face Inference API (optional)

To enable real AI-generated images (instead of the local fallback), the app can use the Hugging Face Inference API.

1. Create a free account at https://huggingface.co/
2. Go to your settings → Access Tokens → New token. Create a token with `read` scope.
3. In GitHub Codespaces, add a secret named `HUGGINGFACE_TOKEN` with that token value.
	- In Codespaces: open the Command Palette → `Codespaces: Add repository secrets` or add it via the repo settings on GitHub (Settings → Secrets → Codespaces).
4. Restart the backend (or reopen the Codespace). The backend will detect `HUGGINGFACE_TOKEN` and route generation requests to the hosted API.

Notes:
- Model used: `runwayml/stable-diffusion-v1-5` via the Hugging Face Inference API.
- Free tier: usually includes a number of free image generations per month (check Hugging Face limits).
- If no token is present, the backend runs in fallback mode and returns a placeholder image so the UI keeps functioning.

