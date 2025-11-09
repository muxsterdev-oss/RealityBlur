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
import VideoOutputNode from './nodes/VideoOutputNode'

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
    type: 'videoOutput',
    position: { x: 800, y: 50 },
    data: { label: 'Video Output', status: 'idle' }
  }
]

const nodeTypes = {
  textInput: TextInputNode,
  aiGenerator: AIGeneratorNode,
  videoOutput: VideoOutputNode
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
    if(runningRef.current) return
    runningRef.current = true
    const sources = nodes.filter(n => !edges.find(e => e.target === n.id))
    for(const s of sources){
      await runNode(s.id)
    }
    runningRef.current = false
  }, [nodes, edges, runNode])

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
