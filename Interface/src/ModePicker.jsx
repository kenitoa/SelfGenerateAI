import React, { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Code2, FileCode2, GitBranch, Settings2, ShieldCheck } from 'lucide-react';
import { normalizeWorkMode, workModes } from './work-modes';
import './mode-picker.css';

export const modeIcons = { code: Code2, review: ShieldCheck, refactor: GitBranch, test: FileCode2 };

export default function ModePicker({ value, onChange, onSettings, disabled }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  const trigger = useRef(null);
  const menu = useRef(null);
  const menuId = React.useId();
  const selected = workModes.find(mode => mode.id === normalizeWorkMode(value));
  const Icon = modeIcons[selected.id];

  useEffect(() => {
    if (!open) return;
    const frame = requestAnimationFrame(() => menu.current?.querySelector('[aria-checked="true"]')?.focus());
    const outside = event => { if (!root.current?.contains(event.target)) setOpen(false); };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('focusin', outside);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('focusin', outside);
    };
  }, [open]);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);

  function choose(id) {
    onChange(id);
    setOpen(false);
    trigger.current?.focus();
  }
  function handleKey(event) {
    if (event.key === 'Escape') {
      event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus();
    }
    const items = [...(menu.current?.querySelectorAll('[role="menuitemradio"], [role="menuitem"]') ?? [])];
    const index = items.indexOf(document.activeElement);
    let next;
    if (event.key === 'ArrowDown') next = (index + 1) % items.length;
    if (event.key === 'ArrowUp') next = (index - 1 + items.length) % items.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = items.length - 1;
    if (next !== undefined && items.length) { event.preventDefault(); items[next].focus(); }
  }

  return <div className="mode-picker" ref={root}>
    <button ref={trigger} type="button" className={`mode-trigger ${open ? 'is-open' : ''}`} disabled={disabled} aria-label={`작업 모드: ${selected.label}`} aria-haspopup="menu" aria-expanded={open} aria-controls={open ? menuId : undefined} onClick={() => setOpen(current => !current)} onKeyDown={event => { if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); setOpen(true); } }}>
      <Icon size={14} /><span>{selected.label}</span><ChevronDown size={12} /><span className="mode-demo">데모</span>
    </button>
    {open && <div className="mode-menu" ref={menu} id={menuId} role="menu" aria-label="작업 모드" onKeyDown={handleKey}>
      <div className="mode-menu-heading" role="presentation"><span>작업 모드</span><small>모델 미연결</small></div>
      {workModes.map(mode => { const ModeIcon = modeIcons[mode.id]; return <button type="button" key={mode.id} role="menuitemradio" aria-checked={value === mode.id} className={`mode-option ${value === mode.id ? 'selected' : ''}`} onClick={() => choose(mode.id)}><ModeIcon size={17} /><span><strong>{mode.label}</strong><small>{mode.description}</small></span>{value === mode.id && <Check size={15} />}</button>; })}
      <div role="separator" className="mode-menu-divider" />
      <button type="button" className="mode-settings" role="menuitem" onClick={() => { trigger.current?.focus(); setOpen(false); onSettings(); }}><Settings2 size={14} />모델 설정</button>
    </div>}
  </div>;
}
