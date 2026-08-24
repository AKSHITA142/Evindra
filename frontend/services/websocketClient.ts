import type { WSEvent, WSEventType } from "@/types/api";

type WSListener = (event: WSEvent) => void;
type WSStatusListener = (connected: boolean) => void;

class WebSocketClient {
  private socket: WebSocket | null = null;
  private jobId: string | null = null;
  private listeners: Map<WSEventType | "all", Set<WSListener>> = new Map();
  private statusListeners: Set<WSStatusListener> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isManualClose = false;

  connect(jobId: string): void {
    // If already connected or currently connecting to the same job, prevent closing/re-opening
    if (
      this.socket &&
      (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING) &&
      this.jobId === jobId
    ) {
      return;
    }

    this.disconnect();
    this.jobId = jobId;
    this.isManualClose = false;
    this.reconnectAttempts = 0;
    this._connect();
  }

  private _connect(): void {
    if (!this.jobId) return;

    const apiPrefix = process.env.NEXT_PUBLIC_API_PREFIX || "/api/v1";
    let wsBase = process.env.NEXT_PUBLIC_WS_URL;

    if (!wsBase) {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
      wsBase = backendUrl.replace(/^http/, "ws");
    }

    // Replace localhost with 127.0.0.1 to avoid macOS IPv6 resolution collision with Docker
    wsBase = wsBase.replace("//localhost", "//127.0.0.1");

    const url = `${wsBase}${apiPrefix}/ws/jobs/${this.jobId}`;

    try {
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        this.reconnectAttempts = 0;
        this._notifyStatus(true);
      };

      this.socket.onmessage = (ev: MessageEvent) => {
        try {
          const raw = JSON.parse(ev.data as string);

          // Normalize the incoming payload into the WSEvent shape the frontend expects.
          const event: WSEvent = {
            event: raw.event,
            job_id: raw.job_id ?? this.jobId ?? "",
            timestamp: raw.timestamp ?? new Date().toISOString(),
            data: {
              status: raw.data?.status ?? raw.status,
              stage: raw.data?.stage ?? raw.stage,
              progress_percent: raw.data?.progress_percent ?? raw.progress_percent ?? raw.progress,
              message: raw.data?.message ?? raw.message,
              level: raw.data?.level ?? raw.level,
              experiment_id: raw.data?.experiment_id ?? raw.experiment_id,
              finding: raw.data?.finding,
              report: raw.data?.report,
            },
          };

          this._dispatch(event);
        } catch {
          // ignore malformed messages
        }
      };

      this.socket.onclose = () => {
        this._notifyStatus(false);
        if (!this.isManualClose) {
          this._scheduleReconnect();
        }
      };

      this.socket.onerror = () => {
        this._notifyStatus(false);
      };
    } catch {
      this._notifyStatus(false);
      this._scheduleReconnect();
    }
  }

  private _scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    this.reconnectTimer = setTimeout(() => this._connect(), delay);
  }

  private _dispatch(event: WSEvent): void {
    // fire specific event listeners
    const specific = this.listeners.get(event.event);
    if (specific) specific.forEach((fn) => fn(event));

    // fire catch-all listeners
    const all = this.listeners.get("all");
    if (all) all.forEach((fn) => fn(event));
  }

  private _notifyStatus(connected: boolean): void {
    this.statusListeners.forEach((fn) => {
      try {
        fn(connected);
      } catch {
        // ignore listener errors
      }
    });
  }

  on(eventType: WSEventType | "all", listener: WSListener): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.get(eventType)?.delete(listener);
    };
  }

  onStatusChange(listener: WSStatusListener): () => void {
    this.statusListeners.add(listener);
    // Emit current status immediately
    listener(this.isConnected);
    return () => {
      this.statusListeners.delete(listener);
    };
  }

  disconnect(): void {
    this.isManualClose = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.socket) {
      this.socket.onopen = null;
      this.socket.onclose = null;
      this.socket.onerror = null;
      this.socket.onmessage = null;
      try {
        this.socket.close();
      } catch {
        // ignore
      }
      this.socket = null;
    }
    this.jobId = null;
    this._notifyStatus(false);
  }

  get isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const wsClient = new WebSocketClient();
