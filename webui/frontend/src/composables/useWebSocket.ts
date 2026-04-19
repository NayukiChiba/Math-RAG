import { onBeforeUnmount, ref } from "vue";
import { wsUrl } from "@/api/http";

export interface WebSocketHandlers<T> {
  onMessage?: (event: T) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
}

export function useWebSocket<T = unknown>(path: string) {
  const socket = ref<WebSocket | null>(null);
  const connected = ref(false);
  let handlers: WebSocketHandlers<T> = {};

  function connect(h: WebSocketHandlers<T> = {}): Promise<WebSocket> {
    disconnect();
    handlers = h;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl(path));
      socket.value = ws;

      ws.onopen = () => {
        connected.value = true;
        handlers.onOpen?.();
        resolve(ws);
      };
      ws.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data) as T;
          handlers.onMessage?.(payload);
        } catch (e) {
          console.warn("WebSocket 消息解析失败:", ev.data, e);
        }
      };
      ws.onerror = (err) => {
        handlers.onError?.(err);
        reject(err);
      };
      ws.onclose = () => {
        connected.value = false;
        socket.value = null;
        handlers.onClose?.();
      };
    });
  }

  function send(data: unknown): void {
    const ws = socket.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(typeof data === "string" ? data : JSON.stringify(data));
    }
  }

  function disconnect(): void {
    const ws = socket.value;
    if (ws) {
      try {
        ws.close();
      } catch {
        /* noop */
      }
    }
    socket.value = null;
    connected.value = false;
  }

  onBeforeUnmount(disconnect);

  return { socket, connected, connect, send, disconnect };
}
