import React from 'react'

export default function FrameSequenceNode({ id, data }){
  return (
    <div style={{ padding: 10, background: '#0f1720', color: 'white', borderRadius: 6, width: 300 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ fontWeight: 'bold' }}>📸 Frame Sequence</div>
        <div style={{ width: 10, height: 10, borderRadius: 6, background: data && data.status === 'running' ? '#ffbf00' : data && data.status === 'success' ? '#00ff88' : data && data.status === 'error' ? '#ff4444' : '#666' }} />
      </div>
      <div style={{ color: '#aaa', fontSize: 13 }}>Collects frames from upstream nodes and prepares sequences.</div>
      {data && data.frames && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, color: '#cfcfcf' }}>Frames: {data.frames.length}</div>
          {data.frames[0] && <img src={data.frames[0]} alt="frame preview" style={{ width: '100%', borderRadius: 6, marginTop: 6 }} />}
        </div>
      )}
    </div>
  )
}
