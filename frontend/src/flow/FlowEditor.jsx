import React, {useCallback, useState, useEffect} from 'react'
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
    data: { label: 'Text Input', prompt: 'A tranquil lake at sunset' }
  },
  {
    id: '2',
    type: 'aiGenerator',
    position: { x: 400, y: 50 },
    data: { label: 'AI Generator', image: null }
  },
  {
    id: '3',
    type: 'videoOutput',
    position: { x: 800, y: 50 },
    data: { label: 'Video Output' }
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

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges])

  // Simple generation runner: called by TextInputNode
  const runGeneration = useCallback(async (prompt) => {
    // call backend /api/generate
    try{
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      })
      const data = await res.json()
      // set the AI node's data to display the result image (node id 2)
      setNodes((nds) => nds.map(n => n.id === '2' ? { ...n, data: { ...n.data, image: data.image, meta: data } } : n))
    }catch(e){
      console.error('Generation failed', e)
    }
  }, [setNodes])

  useEffect(() => {
    const handler = (e) => {
      const { prompt } = e.detail || {}
      if (prompt) runGeneration(prompt)
    }
    window.addEventListener('flow:generate', handler)
    return () => window.removeEventListener('flow:generate', handler)
  }, [runGeneration])

  return (
    <div style={{ height: 520, border: '2px solid #222', borderRadius: 8, padding: 8, background: '#071018' }}>
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
  )
}
