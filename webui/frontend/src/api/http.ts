import axios from "axios";

export const http = axios.create({
  baseURL: "/api",
  timeout: 300000,
});

http.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error?.response?.data?.detail || error?.message || "请求失败";
    return Promise.reject(new Error(detail));
  },
);

/** 构造 WebSocket 地址，自动适配 http/https。 */
export function wsUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}
