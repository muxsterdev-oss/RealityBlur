#!/bin/bash
set -e

echo "🚀 Starting RealityBlur AI Development..."

# Start backend
echo "🖥️  Starting Flask backend on port 5000..."
cd backend && python3 app.py &

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting React frontend on port 3000..."
cd ../frontend && npm run dev &

echo "✅ Development servers starting!"
echo "📁 Backend: http://localhost:5000"
echo "🎨 Frontend: http://localhost:3000"
echo "⚡ Check the ports panel in Codespaces to open your app!"
