import React, { useState } from 'react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);

  const generateVideo = async () => {
    setGenerating(true);
    setResult(null);
    try {
      console.log("🔄 Sending request to backend...");
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      const data = await response.json();
      console.log('✅ Backend response:', data);
      setResult(data);
      
    } catch (error) {
      console.error('❌ Error:', error);
      setResult({ status: "error", message: error.toString() });
    }
    setGenerating(false);
  };

  return (
    <div style={{ 
      padding: '20px', 
      fontFamily: 'Arial, sans-serif',
      background: '#0f0f0f',
      color: 'white',
      minHeight: '100vh'
    }}>
      <h1>🎭 RealityBlur AI</h1>
      <p>Backend Status: <span style={{color: '#00ff88'}}>✅ Connected</span></p>
      
      <div style={{ marginBottom: '20px' }}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe your realistic video... (e.g., 'A person walking in a city at sunset')"
          style={{
            width: '100%',
            height: '80px',
            background: '#1a1a1a',
            border: '1px solid #333',
            color: 'white',
            padding: '10px',
            borderRadius: '4px'
          }}
        />
      </div>

      <button
        onClick={generateVideo}
        disabled={generating}
        style={{
          padding: '12px 24px',
          background: generating ? '#666' : '#00ff88',
          color: 'black',
          border: 'none',
          borderRadius: '6px',
          cursor: generating ? 'not-allowed' : 'pointer',
          fontSize: '16px',
          fontWeight: 'bold'
        }}
      >
        {generating ? '⚡ Generating...' : '🎬 Generate Realistic Video'}
      </button>

      {/* Results Display */}
      {result && (
        <div style={{ 
          marginTop: '30px', 
          padding: '20px', 
          background: result.status === 'success' ? '#1a2a1a' : '#2a1a1a',
          border: `2px solid ${result.status === 'success' ? '#00ff88' : '#ff4444'}`,
          borderRadius: '8px'
        }}>
          <h3>📊 Result:</h3>
          <pre style={{ 
            background: '#2a2a2a', 
            padding: '15px', 
            borderRadius: '4px',
            overflow: 'auto'
          }}>
            {JSON.stringify(result, null, 2)}
          </pre>
          
          {result.image && (
            <div style={{ marginTop: '15px' }}>
              <h4>🖼️ Generated Image:</h4>
              <img 
                src={result.image} 
                style={{ 
                  maxWidth: '512px', 
                  border: '2px solid #333',
                  borderRadius: '8px'
                }} 
                alt="Generated preview" 
              />
            </div>
          )}
        </div>
      )}

      <div style={{ 
        marginTop: '40px', 
        padding: '20px', 
        background: '#1a1a1a',
        borderRadius: '8px'
      }}>
        <h3>🚀 Next Steps:</h3>
        <ul>
          <li>Node-based workflow interface</li>
          <li>Real video generation (not just images)</li>
          <li>Advanced realism enhancements</li>
          <li>Temporal consistency</li>
        </ul>
      </div>
    </div>
  );
}

export default App;
