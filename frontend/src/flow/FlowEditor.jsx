import React, {useCallback, useState, useEffect, useRef} from 'react'
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import TextInputNode from './nodes/TextInputNode'
import AIGeneratorNode from './nodes/AIGeneratorNode'
import FrameSequenceNode from './nodes/FrameSequenceNode'
import VideoOutputNode from './nodes/VideoOutputNode'
import VideoRenderNode from './nodes/VideoRenderNode'

const initialNodes = [
  {
    id: '1',
    type: 'textInput',
    position: { x: 50, y: 50 },
    data: { label: 'Text Input', prompt: 'A tranquil lake at sunset', status: 'idle' }
  },
  {
    id: '2',
    type: 'aiGenerator',
    position: { x: 400, y: 50 },
    data: { label: 'AI Generator', image: null, status: 'idle' }
  },
  {
    id: '3',
    type: 'frameSequence',
    position: { x: 640, y: 50 },
    data: { label: 'Frame Sequence', frames: [], status: 'idle' }
  },
  {
    id: '4',
    type: 'videoOutput',
    position: { x: 980, y: 50 },
    data: { label: 'Video Output', status: 'idle' }
  },
  {
    id: '5',
    type: 'videoRender',
    position: { x: 1200, y: 50 },
    data: { label: 'Video Render', status: 'idle' }
  }
]

const nodeTypes = {
  textInput: TextInputNode,
  aiGenerator: AIGeneratorNode,
  frameSequence: FrameSequenceNode,
  videoOutput: VideoOutputNode,
  videoRender: VideoRenderNode
}

export default function FlowEditor(){
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [rfInstance, setRfInstance] = useState(null)
  const runningRef = useRef(false)

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges])

  // generation helper: call backend and write result into a node
  const runGeneration = useCallback(async (prompt, targetNodeId) => {
    setNodes((nds) => nds.map(n => n.id === targetNodeId ? { ...n, data: { ...n.data, status: 'running' } } : n))
    try{
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      })
      const data = await res.json()
      setNodes((nds) => nds.map(n => n.id === targetNodeId ? { ...n, data: { ...n.data, image: data.image, meta: data, status: 'success' } } : n))
      return data
    }catch(e){
      console.error('Generation failed', e)
      setNodes((nds) => nds.map(n => n.id === targetNodeId ? { ...n, data: { ...n.data, status: 'error', error: String(e) } } : n))
      throw e
    }
  }, [setNodes])

  useEffect(() => {
    const handler = (e) => {
      const { prompt, nodeId } = e.detail || {}
      if (prompt && nodeId) runGeneration(prompt, nodeId)
    }
    const updateHandler = (e) => {
      const { id, data } = e.detail || {}
      if(id){
        setNodes((nds) => nds.map(n => n.id === id ? { ...n, data: { ...n.data, ...data } } : n))
      }
    }
    window.addEventListener('flow:generate', handler)
    window.addEventListener('flow:update-node', updateHandler)
    return () => {
      window.removeEventListener('flow:generate', handler)
      window.removeEventListener('flow:update-node', updateHandler)
    }
  }, [runGeneration, setNodes])

  // Find downstream nodes by following edges
  const getDownstream = useCallback((startId) => {
    const downstream = []
    const queue = [startId]
    const seen = new Set()
    while(queue.length){
      const cur = queue.shift()
      if(seen.has(cur)) continue
      seen.add(cur)
      const outs = edges.filter(e => e.source === cur).map(e => e.target)
      for(const t of outs){
        downstream.push(t)
        queue.push(t)
      }
    }
    return downstream
  }, [edges])

  // Run a single node and propagate outputs to connected nodes
  const runNode = useCallback(async (nodeId) => {
    const node = nodes.find(n => n.id === nodeId)
    if(!node) return
    setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'running' } } : n))

    try{
      if(node.type === 'textInput'){
        const prompt = node.data.prompt || ''
        const downstream = getDownstream(nodeId)
        for(const targetId of downstream){
          const targetNode = nodes.find(n => n.id === targetId)
          if(targetNode && targetNode.type === 'aiGenerator'){
            await runGeneration(prompt, targetId)
          }
        }
      } else if(node.type === 'aiGenerator'){
        let prompt = node.data.prompt
        if(!prompt){
          const incoming = edges.filter(e => e.target === nodeId).map(e => e.source)
          for(const src of incoming){
            const srcNode = nodes.find(n => n.id === src)
            if(srcNode && srcNode.type === 'textInput'){
              prompt = srcNode.data.prompt
              break
            }
          }
        }
        if(prompt){
          await runGeneration(prompt, nodeId)
        } else {
          setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'idle' } } : n))
        }
      } else if(node.type === 'videoOutput'){
        const incoming = edges.filter(e => e.target === nodeId).map(e => e.source)
        const images = []
        for(const src of incoming){
          const srcNode = nodes.find(n => n.id === src)
          if(srcNode && srcNode.data && srcNode.data.image) images.push(srcNode.data.image)
        }
        setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, images, status: 'success' } } : n))
      }
      setNodes((nds) => nds.map(n => n.id === nodeId && n.data.status !== 'success' ? { ...n, data: { ...n.data, status: 'success' } } : n))
    }catch(e){
      setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'error', error: String(e) } } : n))
    }
  }, [nodes, edges, runGeneration, setNodes, getDownstream])

  // Run entire flow starting from source nodes (nodes with no incoming edges)
  const runFlow = useCallback(async () => {
    if (runningRef.current) return
    runningRef.current = true

    // Build adjacency and indegree maps
    const nodeIds = nodes.map(n => n.id)
    const indegree = {}
    const adj = {}
    for (const id of nodeIds) { indegree[id] = 0; adj[id] = [] }
    for (const e of edges) {
      if (adj[e.source]) adj[e.source].push(e.target)
      if (indegree[e.target] === undefined) indegree[e.target] = 0
      indegree[e.target] = (indegree[e.target] || 0) + 1
    }

    // Kahn's algorithm to compute levels (batches of independent nodes)
    const levels = []
    let queue = nodeIds.filter(id => (indegree[id] || 0) === 0)
    const seen = new Set()
    while (queue.length > 0) {
      const thisLevel = [...queue]
      levels.push(thisLevel)
      const nextQueue = []
      for (const u of thisLevel) {
        seen.add(u)
        for (const v of adj[u] || []) {
          indegree[v] = (indegree[v] || 0) - 1
          if (indegree[v] === 0) nextQueue.push(v)
        }
      }
      queue = nextQueue
    }

    // Execute nodes level by level, running nodes in the same level in parallel
      // concurrency controls and retry/backoff helpers
      const maxConcurrent = 3
      let concurrent = 0
      const waiters = []
      const acquire = async () => {
        if (concurrent < maxConcurrent) { concurrent += 1; return () => { concurrent -= 1; const w = waiters.shift(); if (w) w() } }
        await new Promise((resolve) => waiters.push(resolve))
        concurrent += 1
        return () => { concurrent -= 1; const w = waiters.shift(); if (w) w() }
      }

      const sleep = (ms) => new Promise(r => setTimeout(r, ms))
      const runWithRetries = async (fn, attempts = 3) => {
        let lastErr = null
        for (let i = 0; i < attempts; i++) {
          try {
            return await fn()
          } catch (e) {
            lastErr = e
            const backoff = 250 * Math.pow(2, i)
            await sleep(backoff)
          }
        }
        throw lastErr
      }

      for (const level of levels) {
        await Promise.all(level.map(async (nodeId) => {
          const node = nodes.find(n => n.id === nodeId)
          if (!node) return

          // Acquire concurrency slot for potentially network-heavy nodes
          const release = await acquire()
          try {
            if (node.type === 'textInput') {
              setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'success' } } : n))
              return
            }

            if (node.type === 'aiGenerator') {
              let prompt = node.data.prompt
              if (!prompt) {
                const incoming = edges.filter(e => e.target === nodeId).map(e => e.source)
                for (const src of incoming) {
                  const srcNode = nodes.find(n => n.id === src)
                  if (srcNode && srcNode.type === 'textInput') { prompt = srcNode.data.prompt; break }
                }
              }
              if (prompt) {
                await runWithRetries(() => runGeneration(prompt, nodeId), 3)
              } else {
                setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'idle' } } : n))
              }
              return
            }

            if (node.type === 'frameSequence') {
              // collect images from incoming nodes
              const incoming = edges.filter(e => e.target === nodeId).map(e => e.source)
              const frames = []
              for (const src of incoming) {
                const srcNode = nodes.find(n => n.id === src)
                if (srcNode && srcNode.data) {
                  if (srcNode.data.images && srcNode.data.images.length) {
                    frames.push(...srcNode.data.images)
                  } else if (srcNode.data.image) {
                    frames.push(srcNode.data.image)
                  }
                }
              }
              setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, frames, status: frames.length>0 ? 'success' : 'idle' } } : n))
              return
            }

            if (node.type === 'videoOutput') {
              const incoming = edges.filter(e => e.target === nodeId).map(e => e.source)
              const images = []
              for (const src of incoming) {
                const srcNode = nodes.find(n => n.id === src)
                if (srcNode && srcNode.data && srcNode.data.image) images.push(srcNode.data.image)
              }
              setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, images, status: images.length>0 ? 'success' : 'idle' } } : n))
              return
            }

            if (node.type === 'videoRender') {
              // gather frames from incoming frameSequence or generator nodes
              const incoming = edges.filter(e => e.target === nodeId).map(e => e.source)
              const frames = []
              for (const src of incoming) {
                const srcNode = nodes.find(n => n.id === src)
                if (srcNode && srcNode.data) {
                  if (srcNode.data.frames && srcNode.data.frames.length) frames.push(...srcNode.data.frames)
                  else if (srcNode.data.images && srcNode.data.images.length) frames.push(...srcNode.data.images)
                  else if (srcNode.data.image) frames.push(srcNode.data.image)
                }
              }
              if (frames.length === 0) {
                setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'idle' } } : n))
                return
              }

              setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'running' } } : n))
              try {
                const res = await runWithRetries(async () => {
                  const r = await fetch('/api/render-video', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ images: frames, fps: 12 }) })
                  const j = await r.json()
                  if (j.status !== 'success') throw new Error(j.message || 'render failed')
                  return j
                }, 2)
                setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, video: res.video, status: 'success' } } : n))
              } catch (e) {
                setNodes((nds) => nds.map(n => n.id === nodeId ? { ...n, data: { ...n.data, status: 'error', error: String(e) } } : n))
              }
              return
            }
          } finally {
            release()
          }
        }))
      }

    runningRef.current = false
  }, [nodes, edges, runGeneration, setNodes])

  // Run selected node
  const runSelected = useCallback(async (selectedId) => {
    if(!selectedId) return
    await runNode(selectedId)
  }, [runNode])

  // Persistence: save/load via backend
  const saveFlow = useCallback(async (name='flow') => {
    try{
      const payload = { name, nodes, edges }
      await fetch('/api/flows', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      alert('Flow saved')
    }catch(e){
      alert('Save failed: '+e)
    }
  }, [nodes, edges])

  const loadFlow = useCallback(async () => {
    try{
      const res = await fetch('/api/flows')
      const list = await res.json()
      if(!list || list.length===0){ alert('No saved flows'); return }
      const name = list[0]
      const r2 = await fetch(`/api/flows/${encodeURIComponent(name)}`)
      const data = await r2.json()
      if(data.nodes && data.edges){
        setNodes(data.nodes)
        setEdges(data.edges)
      }
    }catch(e){
      alert('Load failed: '+e)
    }
  }, [setNodes, setEdges])

  return (
    <div style={{ padding: 8, border: '2px solid #222', borderRadius: 8, background: '#071018' }}>
      <div style={{ marginBottom: 8, display: 'flex', gap: 8 }}>
        <button onClick={() => runFlow()} style={{ padding: '6px 10px' }}>▶ Run Flow</button>
        <button onClick={() => {
          const sel = nodes.find(n => n.selected)
          runSelected(sel ? sel.id : null)
        }} style={{ padding: '6px 10px' }}>▶ Run Selected</button>
        <button onClick={() => saveFlow(prompt='manual')} style={{ padding: '6px 10px' }}>💾 Save Flow</button>
        <button onClick={() => loadFlow()} style={{ padding: '6px 10px' }}>📂 Load Flow</button>
      </div>
      <div style={{ height: 520 }}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
          />
        </ReactFlowProvider>
      </div>
    </div>
  )
}
