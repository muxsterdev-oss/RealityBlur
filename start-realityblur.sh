#!/bin/bash

echo "🚀 Starting RealityBlur AI Development..."

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: backend or frontend folders not found!"
    echo "📁 Current directory: $(pwd)"
    echo "📁 Contents:"
    ls -la
    exit 1
fi

# Install backend dependencies if needed
echo "📦 Checking backend dependencies..."
cd backend
if [ ! -d "venv" ]; then
    echo "🐍 Setting up Python virtual environment..."
    # prefer python3
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start backend
echo "🖥️  Starting Flask backend on port 5000..."
python app.py &

# Wait for backend to start
sleep 5

# Install frontend dependencies if needed
echo "📦 Checking frontend dependencies..."
cd ../frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start frontend
echo "🌐 Starting React frontend on port 3000..."
npm run dev &

echo "✅ Development servers starting!"
echo "📁 Backend: http://localhost:5000"
echo "🎨 Frontend: http://localhost:3000"
echo "⚡ Check the ports panel in Codespaces to open your app!"
echo "💡 Wait 30-60 seconds for both servers to fully start..."
