#!/bin/bash
set -e
echo "🚀 Setting up RealityBlur AI in Codespaces..."

# Make sure scripts are executable
if [ -f ".devcontainer/setup.sh" ]; then
  chmod +x .devcontainer/setup.sh || true
fi
if [ -f "start-development.sh" ]; then
  chmod +x start-development.sh || true
fi

# Install Python dependencies if backend exists
if [ -d "backend" ]; then
  echo "📦 Installing Python dependencies..."
  cd backend
  if [ -f "requirements.txt" ]; then
    # prefer pip3, fallback to pip
    (python3 -m pip install -r requirements.txt) || (pip install -r requirements.txt) || echo "⚠️ pip install failed — you may need to run it manually or adjust the environment"
  else
    echo "ℹ️ No requirements.txt found in backend"
  fi
  cd - >/dev/null
else
  echo "ℹ️ backend folder not found; skipping Python deps"
fi

# Install Node.js dependencies if frontend exists
if [ -d "frontend" ]; then
  echo "📦 Installing Node.js dependencies..."
  cd frontend
  if [ -f "package.json" ]; then
    npm ci --silent || npm install --silent || echo "⚠️ npm install failed — run it manually in Codespaces"
  else
    echo "ℹ️ No package.json found in frontend"
  fi
  cd - >/dev/null
else
  echo "ℹ️ frontend folder not found; skipping Node deps"
fi

echo "✅ Setup complete!"
echo "🎯 To start development:" 
echo "   Terminal 1: cd backend && python3 app.py"
echo "   Terminal 2: cd frontend && npm run dev"
