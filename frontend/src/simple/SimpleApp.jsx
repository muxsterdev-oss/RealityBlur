import React, {useState} from 'react'

function dataUrlToBlob(dataurl) {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], {type: mime});
}

export default function SimpleApp() {
  const [prompt, setPrompt] = useState('A cinematic scene of a rainy neon city at dusk')
  const [images, setImages] = useState([])
  const [selected, setSelected] = useState([])
  const [videoUrl, setVideoUrl] = useState(null)
  const [modelsUsed, setModelsUsed] = useState([])
  const [hfDiagnostics, setHfDiagnostics] = useState(null)
  const [loading, setLoading] = useState(false)

  async function generate() {
    setLoading(true)
    try {
      const res = await fetch('/api/generate-variations', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt, num_variations: 6})
      })
      const j = await res.json()
      if (j.status === 'success' && Array.isArray(j.images)) {
        setImages(j.images)
        setSelected([])
        setVideoUrl(null)
        setModelsUsed(Array.isArray(j.models_used) ? j.models_used : [])
        setHfDiagnostics(j.hf_diagnostics || null)
      } else {
        alert('Generation failed: ' + (j.message || JSON.stringify(j)))
      }
    } catch (e) {
      alert('Generation error: ' + e)
    }
    setLoading(false)
  }

  function toggleSelect(idx) {
    setSelected(s => s.includes(idx) ? s.filter(x => x!==idx) : [...s, idx])
  }

  async function renderVideo() {
    if (!selected.length) { alert('Select at least one image'); return }
    setLoading(true)
    try {
      const chosen = selected.map(i => images[i])
      // Build a social-media-optimized payload (TikTok vertical by default)
      const defaultEffects = chosen.map((_, idx) => ({
        type: 'ken_burns',
        zoom: idx % 2 === 0 ? 1.3 : 1.2,
        pan_x: idx % 2 === 0 ? 0.1 : -0.1,
        pan_y: 0
      }))
      const defaultTransitions = []
      for (let i = 0; i < Math.max(0, chosen.length - 1); i++) defaultTransitions.push(i % 2 === 0 ? 'slide_left' : 'fade')

      const payload = {
        images: chosen,
        preset: 'tiktok',
        aspect_ratio: '9:16',
        fps: 30,
        duration: Math.max(3, chosen.length * 2),
        effects: defaultEffects,
        transitions: defaultTransitions,
        motion_blur: true,
        film_grain: 0.04,
        color_grade: 'cinematic',
        sharpening: 0.6,
        variable_timing: true
      }

      const res = await fetch('/api/render-dynamic-video', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      })
      const j = await res.json()
      if (j.status === 'success' && j.video) {
        setVideoUrl(j.video)
      } else {
        alert('Video render failed: ' + (j.message || JSON.stringify(j)))
      }
    } catch (e) {
      alert('Render error: ' + e)
    }
    setLoading(false)
  }

  return (
    <div style={{padding:20,fontFamily:'system-ui, Arial'}}>
      <h2>RealityBlur — Simple Creator</h2>
      <div style={{marginBottom:12}}>
        <textarea value={prompt} onChange={e=>setPrompt(e.target.value)} rows={3} style={{width:'100%'}} />
        <button onClick={generate} disabled={loading} style={{marginTop:8}}>Generate Variations</button>
      </div>

      <div>
        <h3>Gallery</h3>
        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(160px, 1fr))', gap:8}}>
          {images.map((src, i) => (
            <div key={i} style={{border: selected.includes(i) ? '3px solid #0af' : '1px solid #ddd', padding:6}}>
              <img src={src} alt={`v${i}`} style={{width:'100%',height:160,objectFit:'cover'}} />
                  {modelsUsed && modelsUsed[i] && (
                    <div style={{fontSize:11, color:'#444', marginTop:6}}>Model: {modelsUsed[i]}</div>
                  )}
              <div style={{display:'flex',justifyContent:'space-between',marginTop:6}}>
                <label><input type='checkbox' checked={selected.includes(i)} onChange={()=>toggleSelect(i)} /> Select</label>
                <a download={`image_${i}.png`} href={src}>Download</a>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{marginTop:16}}>
        <h3>Render</h3>
        <button onClick={renderVideo} disabled={loading || !selected.length}>Render Video from Selected</button>
        {videoUrl && (
          <div style={{marginTop:12}}>
            <video controls src={videoUrl} style={{maxWidth:'100%'}} />
            <div><a href={videoUrl} download='realityblur.mp4'>Download Video</a></div>
          </div>
        )}
      </div>

      {hfDiagnostics && (
        <div style={{marginTop:12, padding:8, background:'#fff3f0', border:'1px solid #ffd0c2'}}>
          <strong>Hugging Face diagnostics:</strong>
          <pre style={{whiteSpace:'pre-wrap', fontSize:12, marginTop:6}}>{hfDiagnostics}</pre>
        </div>
      )}

      {modelsUsed && modelsUsed.length>0 && (
        <div style={{marginTop:12, fontSize:13, color:'#333'}}>First model used: <strong>{modelsUsed[0]}</strong></div>
      )}

      {loading && <div style={{marginTop:12}}>Working…</div>}
    </div>
  )
}
