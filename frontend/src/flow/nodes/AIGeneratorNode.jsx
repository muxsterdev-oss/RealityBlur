import React, { useEffect } from 'react'

export default function AIGeneratorNode({ id, data }){
  // data.image is expected to be a data-url string
  return (
    <div style={{ padding: 10, background: '#1a1a2a', color: 'white', borderRadius: 6, width: 300 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ fontWeight: 'bold' }}>🖼️ AI Generator</div>
        <div style={{ width: 10, height: 10, borderRadius: 6, background: data.status === 'running' ? '#ffbf00' : data.status === 'success' ? '#00ff88' : data.status === 'error' ? '#ff4444' : '#666' }} />
      </div>
      {data && data.meta && (
            <div style={{ fontSize: 12, marginBottom: 6, color: '#cfcfcf' }}>
              Source: {data.meta.source} • {data.meta.generation_time_ms} ms
              {data.meta.hf_diagnostics ? (
                <div style={{ marginTop: 4, color: '#ffcccb', fontSize: 11 }}>HF: {data.meta.hf_diagnostics}</div>
              ) : null}
            </div>
      )}
      {data && data.image ? (
        <img src={data.image} alt="preview" style={{ width: '100%', borderRadius: 6, border: '1px solid #333' }} />
      ) : (
        <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>No preview</div>
      )}
    </div>
  )
}
