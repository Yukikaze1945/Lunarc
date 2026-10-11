import { fetchQueue, wsUrl } from './api'
import { QueueFeed } from './queueFeed'

const feed = new QueueFeed(signal => fetchQueue(signal), () => new WebSocket(wsUrl('queue')))
export const subscribeQueue = feed.subscribe.bind(feed)
export const queueSnapshot = feed.getSnapshot.bind(feed)
export async function refreshQueue() {
  const state = await feed.refresh()
  if (!state) throw new Error('暂时无法读取导出队列，请稍后重试')
  return state
}
