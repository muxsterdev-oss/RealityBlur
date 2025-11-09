import React, { useEffect } from 'react'

export default function AIGeneratorNode({ id, data }){
  // data.image is expected to be a data-url string
  return (
    <div style={{ padding: 10, background: '#1a1a2a', color: 'white', borderRadius: 6, width: 300 }}>
      <div style={{ fontWeight: 'bold', marginBottom: 6 }}>🖼️ AI Generator</div>
      {data && data.meta && (
        <div style={{ fontSize: 12, marginBottom: 6, color: '#cfcfcf' }}>Source: {data.meta.source} • {data.meta.generation_time_ms} ms</div>
      )}
      {data && data.image ? (
        <img src={data.image} alt="preview" style={{ width: '100%', borderRadius: 6, border: '1px solid #333' }} />
      ) : (
        <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>No preview</div>
      )}
    </div>
  )
}
