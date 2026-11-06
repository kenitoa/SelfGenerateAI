import { useCallback, useEffect, useRef, useState } from 'react';
import { datasetRecord, items, normalizeThread, ragRequest, readSession } from './rag-api';

export const defaultSettings = { name: '나의 워크스페이스', endpoint: 'http://localhost:11434', model: 'qwen2.5-coder:0.5b', memory: '2', schedule: '02:00', auto: false };
export const defaultTraining = { endpoint: '', epochs: '3', batch: '1', rate: '0.0002' };

export default function useRag(period) {
  const [records, setRecords] = useState([]);
  const [sources, setSources] = useState([]);
  const [threads, setThreads] = useState([]);
  const [settings, setSettings] = useState(defaultSettings);
  const [trainingConfig, setTrainingConfig] = useState(defaultTraining);
  const [settingsVersion, setSettingsVersion] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [system, setSystem] = useState(null);
  const [activities, setActivities] = useState([]);
  const [capabilities, setCapabilities] = useState({});
  const [auth, setAuth] = useState(null);
  const [connection, setConnection] = useState('connecting');
  const [error, setError] = useState('');
  const [lastReceivedAt, setLastReceivedAt] = useState(null);
  const loadedSettings = useRef(false);
  const inFlight = useRef(false);
  const alive = useRef(true);
  const periodRef = useRef(period);
  periodRef.current = period;
  useEffect(() => {
    const invalidate = () => { loadedSettings.current = false; };
    window.addEventListener('seed-llm-settings', invalidate);
    return () => window.removeEventListener('seed-llm-settings', invalidate);
  }, []);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const session = await readSession();
      if (!alive.current) return;
      setAuth(session);
      if (session.authenticated === false) { setConnection('unauthenticated'); return; }
      const paths = ['/datasets', '/sources', '/threads', `/dashboard?period=${periodRef.current === '월간' ? 'month' : 'week'}`, '/system/status', '/activities', '/capabilities'];
      if (!loadedSettings.current) paths.push('/settings');
      const responses = await Promise.allSettled(paths.map(path => ragRequest(path)));
      if (!alive.current) return;
      const update = [v => setRecords(items(v).map(datasetRecord)), v => setSources(items(v)), v => setThreads(items(v).map(normalizeThread)), setDashboard, setSystem, v => setActivities(items(v)), setCapabilities, value => {
        const config = value.settings ?? value;
        setSettings(old => ({ ...old, ...config }));
        setTrainingConfig({ ...defaultTraining, ...(config.training ?? config.trainingConfig ?? {}) });
        setSettingsVersion(value.version ?? config.version ?? 1); loadedSettings.current = true;
      }];
      let firstError;
      responses.forEach((result, index) => { if (result.status === 'fulfilled') update[index](result.value); else firstError ??= result.reason; });
      const essential = responses[0].status === 'fulfilled' && responses[3].status === 'fulfilled';
      if (essential) { setLastReceivedAt(new Date().toISOString()); setConnection(firstError ? 'stale' : 'connected'); setError(firstError?.message ?? ''); }
      else { setConnection(old => old === 'connected' || old === 'stale' ? 'stale' : 'disconnected'); setError(firstError?.message ?? 'RAG 서버 연결 실패'); }
    } catch (cause) {
      if (alive.current) { setConnection(old => old === 'connected' || old === 'stale' ? 'stale' : 'disconnected'); setError(cause.message ?? 'RAG 서버 연결 실패'); }
    } finally { inFlight.current = false; }
  }, []);

  useEffect(() => {
    alive.current = true;
    refresh();
    const timer = setInterval(() => { if (!document.hidden) refresh(); }, 5000);
    const onVisible = () => { if (!document.hidden) refresh(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => { alive.current = false; clearInterval(timer); document.removeEventListener('visibilitychange', onVisible); };
  }, [refresh]);
  useEffect(() => { refresh(); }, [period, refresh]);

  async function saveSettings(patch = {}) {
    let payload;
    try { payload = await ragRequest('/settings', { method: 'PATCH', body: { ...settings, training: trainingConfig, ...patch }, headers: settingsVersion === null ? {} : { 'If-Match': String(settingsVersion) } }); }
    catch (error) {
      if (error.status === 412) {
        const latest = await ragRequest('/settings');
        const config = latest.settings ?? latest;
        setSettings({ ...defaultSettings, ...config }); setTrainingConfig({ ...defaultTraining, ...(config.training ?? {}) });
        setSettingsVersion(latest.version ?? config.version);
        error.message = '설정 충돌 · 서버 설정을 다시 불러왔습니다. 확인 후 저장하세요.';
      }
      throw error;
    }
    const next = payload.settings ?? payload;
    setSettings(old => ({ ...old, ...next }));
    if (next.training) setTrainingConfig({ ...defaultTraining, ...next.training });
    setSettingsVersion(payload.version ?? next.version ?? settingsVersion);
    return payload;
  }
  async function loadThread(id) {
    const thread = normalizeThread(await ragRequest(`/threads/${encodeURIComponent(id)}`));
    setThreads(previous => [thread, ...previous.filter(item => item.id !== id)]);
    return thread;
  }
  return { records, sources, threads, setThreads, settings, setSettings, trainingConfig, setTrainingConfig, dashboard, system, activities, capabilities, auth, connection, error, lastReceivedAt, refresh, saveSettings, loadThread };
}
