import React from 'react'

export default function VideoOutputNode({ data }){
  return (
    <div style={{ padding: 10, background: '#221a2a', color: 'white', borderRadius: 6, width: 260 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ fontWeight: 'bold' }}>🎞️ Video Output</div>
        <div style={{ width: 10, height: 10, borderRadius: 6, background: data && data.status === 'running' ? '#ffbf00' : data && data.status === 'success' ? '#00ff88' : data && data.status === 'error' ? '#ff4444' : '#666' }} />
      </div>
      <div style={{ color: '#aaa', fontSize: 13 }}>Preview & export node (placeholder)</div>
      {data && data.images && data.images.length>0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, color: '#cfcfcf' }}>Images: {data.images.length}</div>
          <img src={data.images[0]} style={{ width: '100%', borderRadius: 6, marginTop: 6 }} />
        </div>
      )}
    </div>
  )
}
