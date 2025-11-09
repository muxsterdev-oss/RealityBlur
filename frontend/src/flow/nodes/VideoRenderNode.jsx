import React from 'react'

export default function VideoRenderNode({ id, data }){
  return (
    <div style={{ padding: 10, background: '#071022', color: 'white', borderRadius: 6, width: 300 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ fontWeight: 'bold' }}>🎬 Video Render</div>
        <div style={{ width: 10, height: 10, borderRadius: 6, background: data && data.status === 'running' ? '#ffbf00' : data && data.status === 'success' ? '#00ff88' : data && data.status === 'error' ? '#ff4444' : '#666' }} />
      </div>
      <div style={{ color: '#aaa', fontSize: 13 }}>Renders a sequence of frames into an MP4 using backend FFmpeg.</div>
      {data && data.video && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, color: '#cfcfcf' }}>Rendered video</div>
          <video src={data.video} controls style={{ width: '100%', borderRadius: 6, marginTop: 6 }} />
        </div>
      )}
    </div>
  )
}
