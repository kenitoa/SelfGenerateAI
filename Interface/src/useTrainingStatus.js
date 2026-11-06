import { useCallback, useEffect, useRef, useState } from 'react';
import { emptyTrainingStatus, trainingRequest } from './training-api';

export default function useTrainingStatus(endpoint) {
  const [status, setStatus] = useState(emptyTrainingStatus);
  const [connection, setConnection] = useState('disconnected');
  const [lastReceivedAt, setLastReceivedAt] = useState(null);
  const [error, setError] = useState('');
  const [commandError, setCommandError] = useState('');
  const [pending, setPending] = useState(false);
  const refreshRef = useRef(() => {});
  const generation = useRef(0);
  const commandController = useRef(null);
  const commandBusy = useRef(false);

  useEffect(() => {
    const current = ++generation.current;
    let disposed = false, timer, controller, received = false, inFlight = false;
    commandController.current?.abort();
    setPending(false); setStatus(emptyTrainingStatus()); setLastReceivedAt(null); setError(''); setCommandError('');
    setConnection(endpoint ? 'connecting' : 'disconnected');
    async function poll() {
      if (disposed || !endpoint || inFlight) return;
      clearTimeout(timer);
      inFlight = true;
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      let succeeded = false;
      try {
        const next = await trainingRequest(endpoint, 'status', { signal: controller.signal });
        if (disposed || current !== generation.current) return;
        setStatus(next); setLastReceivedAt(new Date().toISOString()); setConnection('connected'); setError('');
        received = true; succeeded = true;
      } catch (cause) {
        if (disposed || current !== generation.current) return;
        setConnection(received ? 'stale' : 'disconnected');
        setError(cause.name === 'AbortError' ? '학습 서버 응답 시간 초과' : cause.message || '학습 서버 연결 실패');
      } finally {
        clearTimeout(timeout); inFlight = false;
        if (!disposed) timer = setTimeout(poll, succeeded ? 1000 : 5000);
      }
    }
    refreshRef.current = poll;
    if (endpoint) poll();
    return () => { disposed = true; clearTimeout(timer); controller?.abort(); commandController.current?.abort(); refreshRef.current = () => {}; };
  }, [endpoint]);

  const refresh = useCallback(() => refreshRef.current(), []);
  async function command(action, body) {
    if (connection !== 'connected' || commandBusy.current) return;
    commandBusy.current = true; setPending(true); setCommandError('');
    const current = generation.current;
    const controller = new AbortController(); commandController.current = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    const path = action === 'start' ? 'jobs' : `jobs/${encodeURIComponent(status.job.id)}/${action}`;
    try {
      await trainingRequest(endpoint, path, { method: 'POST', body, signal: controller.signal });
      if (current === generation.current) { setError(''); await refreshRef.current(); }
    } catch (cause) {
      if (current === generation.current) {
        setCommandError(cause.name === 'AbortError' ? '요청 결과 미확인 · 서버 상태를 확인하세요.' : cause.message || '학습 요청 실패');
        await refreshRef.current();
      }
    } finally {
      clearTimeout(timeout); commandBusy.current = false;
      if (current === generation.current) setPending(false);
    }
  }
  return { status, connection, lastReceivedAt, error: commandError || error, pending, refresh, command };
}
