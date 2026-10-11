import { useLayoutEffect, useRef, useState } from 'react'
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import type { RenderStage } from '../types'
import { moveStage } from '../services/renderPipeline'

type Drag = {
  stage: RenderStage; pointer: number; startY: number; listTop: number
  order: RenderStage[]; rects: DOMRect[]; delta: number; target: number; active: boolean
}

/** Keep the DOM order stable during a gesture; commit only on release. */
export function useStageSort(stages: RenderStage[], save: (order: RenderStage[]) => void) {
  const nodes = useRef(new Map<RenderStage, HTMLLIElement>())
  const list = useRef<HTMLOListElement>(null)
  const drag = useRef<Drag | null>(null)
  const [view, setView] = useState<Drag | null>(null)
  const pending = useRef<Map<RenderStage, DOMRect> | null>(null)
  const animations = useRef<Animation[]>([])
  const snapshot = () => new Map([...nodes.current].map(([key, node]) => [key, node.getBoundingClientRect()]))
  const animateSave = (order: RenderStage[]) => {
    pending.current = snapshot()
    save(order)
  }
  useLayoutEffect(() => {
    const previous = pending.current
    if (!previous) return
    pending.current = null
    animations.current.forEach(animation => animation.cancel())
    animations.current = []
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    previous.forEach((rect, stage) => {
      const node = nodes.current.get(stage)
      if (!node) return
      const offset = rect.top - node.getBoundingClientRect().top
      if (Math.abs(offset) > .5) animations.current.push(node.animate(
        [{ transform: `translateY(${offset}px)` }, { transform: 'translateY(0)' }],
        { duration: 200, easing: 'cubic-bezier(.2,.8,.2,1)' },
      ))
    })
  })
  useLayoutEffect(() => () => animations.current.forEach(animation => animation.cancel()), [])
  const orderKey = stages.join(',')
  useLayoutEffect(() => {
    if (drag.current && drag.current.order.join(',') !== orderKey) {
      drag.current = null
      setView(null)
    }
  }, [orderKey])

  const finish = (commit: boolean) => {
    const current = drag.current
    if (!current) return
    if (current.active) pending.current = snapshot()
    drag.current = null
    setView(null)
    if (commit && current.active && current.target !== current.order.indexOf(current.stage)) {
      save(moveStage(current.order, current.order.indexOf(current.stage), current.target))
    }
  }
  const handle = (stage: RenderStage) => ({
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0 || drag.current) return
      animations.current.forEach(animation => animation.cancel())
      const rects = stages.map(key => nodes.current.get(key)!.getBoundingClientRect())
      drag.current = { stage, pointer: event.pointerId, startY: event.clientY,
        listTop: list.current!.getBoundingClientRect().top, order: [...stages], rects,
        delta: 0, target: stages.indexOf(stage), active: false }
      event.currentTarget.setPointerCapture(event.pointerId)
      event.preventDefault()
      event.currentTarget.focus()
    },
    onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>) => {
      const current = drag.current
      if (!current || current.pointer !== event.pointerId) return
      // Compensate for scrolling while the pointer is captured.
      const delta = event.clientY - current.startY - (list.current!.getBoundingClientRect().top - current.listTop)
      if (!current.active && Math.abs(delta) < 5) return
      const index = current.order.indexOf(stage)
      const center = current.rects[index].top + current.rects[index].height / 2 + delta
      let target = index
      for (let i = 0; i < current.rects.length; i++) {
        const midpoint = current.rects[i].top + current.rects[i].height / 2
        if (i < index && center < midpoint) { target = i; break }
        if (i > index && center > midpoint) target = i
      }
      drag.current = { ...current, delta, target, active: true }
      setView(drag.current)
    },
    onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => {
      if (drag.current?.pointer === event.pointerId) finish(true)
    },
    onPointerCancel: () => finish(false),
    onLostPointerCapture: () => finish(false),
    onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === 'Escape') { finish(false); event.preventDefault() }
      else if (!drag.current && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
        event.preventDefault()
        const index = stages.indexOf(stage)
        animateSave(moveStage(stages, index, index + (event.key === 'ArrowUp' ? -1 : 1)))
      }
    },
  })
  const style = (stage: RenderStage) => {
    if (!view?.active) return undefined
    if (view.stage === stage) return { transform: `translateY(${view.delta}px)`, zIndex: 2 }
    const from = view.order.indexOf(view.stage)
    const reordered = moveStage(view.order, from, view.target)
    const gap = view.rects.length > 1 ? view.rects[1].top - view.rects[0].bottom : 0
    let top = view.rects[0].top
    for (const key of reordered) {
      if (key === stage) break
      top += view.rects[view.order.indexOf(key)].height + gap
    }
    return { transform: `translateY(${top - view.rects[view.order.indexOf(stage)].top}px)` }
  }
  return { list, nodes, handle, style, animateSave, active: view?.active ? view.stage : null }
}
