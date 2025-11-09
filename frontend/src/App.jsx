import React, { useState, useEffect } from 'react';
import FlowEditor from './flow/FlowEditor'

function App() {
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [apiAvailable, setApiAvailable] = useState(false);

  // On mount, check backend health to determine whether hosted API is available
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const res = await fetch('/api/health');
        const j = await res.json();
        if (mounted) setApiAvailable(Boolean(j.api_available));
      } catch (e) {
        if (mounted) setApiAvailable(false);
      }
    };
    check();
    // optionally poll health every 30s
    const id = setInterval(check, 30000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

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
      <p>
        Backend Status: <span style={{color: '#00ff88'}}>✅ Connected</span>
        {' \u00A0'}
        {apiAvailable ? (
          <span style={{color: '#00ff88', fontWeight: 'bold'}}>🟢 AI Mode: Using Hugging Face API</span>
        ) : (
          <span style={{color: '#ff4444', fontWeight: 'bold'}}>🔴 Fallback Mode: Add HuggingFace Token for Real AI</span>
        )}
      </p>
      
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
        {generating ? '⏳ Generating AI Image...' : '🎬 Generate Realistic Image'}
      </button>

      {generating && (
        <div style={{ marginTop: '12px', color: '#00ff88' }}>⏳ Generating AI Image... this may take a moment</div>
      )}

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

          {/* Source label and generation time */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            {result.source === 'huggingface' ? (
              <span style={{ color: '#00ff88', fontWeight: 'bold' }}>🟢 AI Generated</span>
            ) : result.source === 'local' ? (
              <span style={{ color: '#00aaff', fontWeight: 'bold' }}>🟣 Local Model</span>
            ) : (
              <span style={{ color: '#ff4444', fontWeight: 'bold' }}>🔴 Fallback Image</span>
            )}

            {typeof result.generation_time_ms !== 'undefined' && (
              <span style={{ color: '#cfcfcf' }}>⏱️ {result.generation_time_ms} ms</span>
            )}
          </div>

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

      <div style={{ marginTop: 20 }}>
        <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} style={{ marginRight: 8 }}>Back to Top</button>
      </div>

      <div style={{ marginTop: 30 }}>
        <h2 style={{ marginBottom: 10 }}>🧩 Workflow Editor (Beta)</h2>
        <FlowEditor />
      </div>
    </div>
  );
}

export default App;
