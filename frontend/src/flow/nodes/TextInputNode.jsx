import React, { useState } from 'react'

export default function TextInputNode({ id, data }){
  const [value, setValue] = useState(data.prompt || '')

  // React Flow will pass custom props; we expose a global event on window for simplicity
  const onGenerate = () => {
    const event = new CustomEvent('flow:generate', { detail: { prompt: value, nodeId: id } })
    window.dispatchEvent(event)
  }

  return (
    <div style={{ padding: 10, background: '#0b2a2a', color: 'white', borderRadius: 6, width: 220 }}>
      <div style={{ fontWeight: 'bold', marginBottom: 6 }}>📝 Text Input</div>
      <textarea value={value} onChange={e=>setValue(e.target.value)} style={{ width: '100%', height: 80, borderRadius: 4 }} />
      <button onClick={onGenerate} style={{ marginTop: 8, width: '100%', padding: 6, background: '#00bfa5', border: 'none', borderRadius: 4 }}>Generate</button>
    </div>
  )
}
