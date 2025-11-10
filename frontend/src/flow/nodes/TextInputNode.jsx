import React, { useState, useEffect } from 'react'

export default function TextInputNode({ id, data }){
  const [value, setValue] = useState(data.prompt || '')

  // When value changes, notify the FlowEditor so it can persist into node.data
  useEffect(() => {
    const ev = new CustomEvent('flow:update-node', { detail: { id, data: { prompt: value } } })
    window.dispatchEvent(ev)
  }, [value, id])

  const onGenerate = () => {
    const event = new CustomEvent('flow:generate', { detail: { prompt: value, nodeId: id } })
    window.dispatchEvent(event)
  }

  return (
    <div style={{ padding: 10, background: '#0b2a2a', color: 'white', borderRadius: 6, width: 220 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ fontWeight: 'bold' }}>📝 Text Input</div>
        <div style={{ width: 10, height: 10, borderRadius: 6, background: data.status === 'running' ? '#ffbf00' : data.status === 'success' ? '#00ff88' : data.status === 'error' ? '#ff4444' : '#666' }} />
      </div>
      <textarea value={value} onChange={(e)=>setValue(e.target.value)} style={{ width: '100%', height: 80, borderRadius: 4 }} />
      <button onClick={onGenerate} style={{ marginTop: 8, width: '100%', padding: 6, background: '#00bfa5', border: 'none', borderRadius: 4 }}>Generate</button>
    </div>
  )
}
