"""Text-only ingestion. Source code is data and is never executed or corrected."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import PurePath

from RAG.core import DomainError

ALLOWED = {'.md', '.txt', '.json', '.jsonl', '.py', '.js', '.jsx', '.ts', '.tsx', '.sql', '.html', '.css', '.yml', '.yaml', '.toml'}
ALLOWED.update({'.rs','.go','.c','.cpp','.cc','.h','.hpp','.cs','.java','.kt','.kts','.swift','.m','.mm','.rb','.php','.sh','.bash','.ps1','.r','.jl','.lua','.dart','.scala','.ex','.exs','.erl','.hrl','.hs','.clj','.cljs','.vue','.svelte','.graphql','.gql','.proto','.zig','.nim','.pl','.pm','.f90','.f95','.lisp','.el','.scm','.rkt','.ml','.mli','.fs','.fsx','.vb','.v','.sv','.vhd','.vhdl','.asm','.s','.bat','.cmd','.groovy','.gradle','.sol','.move','.nix','.tf','.hcl','.cmake','.make'})
LANGUAGES = {'.py': 'python', '.js': 'javascript', '.jsx': 'javascript', '.ts': 'typescript', '.tsx': 'typescript', '.sql': 'sql', '.html': 'html', '.css': 'css', '.json': 'json', '.jsonl': 'json', '.md': 'markdown', '.yml': 'yaml', '.yaml': 'yaml', '.toml': 'toml'}
LANGUAGES.update({'.rs':'rust','.go':'go','.c':'c','.cpp':'cpp','.h':'c','.hpp':'cpp','.cs':'csharp','.java':'java','.kt':'kotlin','.swift':'swift','.rb':'ruby','.php':'php','.sh':'shell','.ps1':'powershell','.r':'r','.jl':'julia','.lua':'lua','.dart':'dart','.scala':'scala','.ex':'elixir','.hs':'haskell','.clj':'clojure','.vue':'vue','.svelte':'svelte','.zig':'zig','.sol':'solidity'})
LANGUAGES.update({'.cc':'cpp','.kts':'kotlin','.bash':'shell','.exs':'elixir','.cljs':'clojure'})
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{20,}\b'),
    re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*["']([^"'\n]{16,})["']'''),
]
INJECTION = re.compile(r'(?i)ignore (?:all |the )?(?:previous|prior) instructions|reveal (?:the )?system prompt|이전\s*(?:지시|명령).*무시')


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def logical_name(name: str) -> str:
    name = unicodedata.normalize('NFC', str(name).replace('\\', '/'))
    if not name or len(name) > 512 or '\x00' in name or any(part in {'.', '..'} for part in name.split('/')) or name.startswith('/') or re.match(r'^[A-Za-z]:', name):
        raise DomainError('INVALID_FILENAME', '상대 경로 파일 이름이 필요합니다.', 422)
    return name


def decode(raw: bytes) -> str:
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise DomainError('INVALID_ENCODING', 'UTF-8 텍스트 파일이 필요합니다.', 422, {'offset': exc.start}) from exc
    if '\x00' in text or any(ord(ch) < 32 and ch not in '\n\r\t\f' for ch in text):
        raise DomainError('BINARY_NOT_ALLOWED', '바이너리 파일은 지원하지 않습니다.', 422)
    if not text.strip():
        raise DomainError('EMPTY_DOCUMENT', '빈 파일은 추가할 수 없습니다.', 422)
    return text


def depth(value, current=0):
    if current > 32:
        raise DomainError('JSON_TOO_DEEP', 'JSON 깊이는 최대 32입니다.', 422)
    if isinstance(value, dict):
        for entry in value.values():
            depth(entry, current + 1)
    elif isinstance(value, list):
        for entry in value:
            depth(entry, current + 1)


def record_text(row: dict) -> str:
    for key in ('content', 'text', 'code'):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    if isinstance(row.get('instruction'), str) and isinstance(row.get('output'), str):
        return f"Instruction:\n{row['instruction']}\n\nOutput:\n{row['output']}"
    if isinstance(row.get('messages'), list):
        result = []
        for item in row['messages']:
            if not isinstance(item, dict) or item.get('role') not in {'user', 'assistant', 'system'} or not isinstance(item.get('content'), str):
                raise DomainError('INVALID_MESSAGES', 'messages에는 role과 content가 필요합니다.', 422)
            result.append(f"{item['role']}:\n{item['content']}")
        if result:
            return '\n\n'.join(result)
    raise DomainError('INVALID_RECORD', '레코드에 content, text, code, instruction/output 또는 messages가 필요합니다.', 422)


def validate_file(name: str, raw: bytes, *, allow_json_document=False) -> dict:
    name = logical_name(name)
    ext = PurePath(name).suffix.lower()
    if ext not in ALLOWED:
        raise DomainError('UNSUPPORTED_FILE', '지원하지 않는 파일 형식입니다.', 422, {'filename': name})
    if len(raw) > 10 * 1024 * 1024:
        raise DomainError('UPLOAD_TOO_LARGE', '파일당 최대 10 MiB입니다.', 413, {'filename': name})
    text = decode(raw)
    records = []
    if ext == '.json' and allow_json_document:
        try:
            parsed = json.loads(text)
            depth(parsed)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise DomainError('INVALID_JSON', '수집 JSON 문서 형식이 올바르지 않습니다.', 422, {'filename': name, 'line': getattr(exc, 'lineno', None)}) from exc
        return {'name': name, 'raw': raw, 'text': text, 'hash': digest(raw), 'ext': ext, 'records': [{'key': '', 'text': text, 'recordHash': digest(text), 'positions': [], 'language': 'json', 'title': name}], 'oversized': len(raw) > 2 * 1024 * 1024}
    if ext in {'.json', '.jsonl'}:
        try:
            if ext == '.jsonl':
                rows = [(line_no, json.loads(line)) for line_no, line in enumerate(text.splitlines(), 1) if line.strip()]
            else:
                parsed = json.loads(text)
                if not isinstance(parsed, list):
                    raise DomainError('INVALID_JSON', 'JSON 파일은 객체 배열이어야 합니다.', 422)
                rows = list(enumerate(parsed, 1))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise DomainError('INVALID_JSON', 'JSON 형식이 올바르지 않습니다.', 422, {'filename': name, 'line': getattr(exc, 'lineno', None)}) from exc
        if not rows or len(rows) > 10000:
            raise DomainError('INVALID_RECORD_COUNT', '파일당 1~10,000 레코드가 필요합니다.', 422)
        seen = {}
        for position, row in rows:
            if not isinstance(row, dict):
                raise DomainError('INVALID_RECORD', '각 JSON 레코드는 객체여야 합니다.', 422, {'record': position})
            depth(row)
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            if len(serialized.encode('utf-8')) > 65536:
                raise DomainError('RECORD_TOO_LARGE', '레코드당 최대 64 KiB입니다.', 422, {'record': position})
            extracted = record_text(row)
            key = str(row.get('id', '')).strip() or digest(serialized)
            if len(key) > 512:
                raise DomainError('INVALID_RECORD_ID', '레코드 ID가 너무 깁니다.', 422)
            if key in seen:
                old = seen[key]
                if old['recordHash'] != digest(serialized):
                    raise DomainError('DUPLICATE_RECORD_ID', '서로 다른 레코드가 같은 ID를 사용합니다.', 422)
                old['positions'].append(position)
                continue
            record = {'key': key, 'text': extracted, 'rawRecord': serialized, 'recordHash': digest(serialized), 'positions': [position], 'language': str(row.get('language', LANGUAGES[ext])).lower(), 'title': str(row.get('title') or f'{name} · {position}')[:300]}
            seen[key] = record
            records.append(record)
    else:
        records = [{'key': '', 'text': text, 'recordHash': digest(text), 'positions': [], 'language': LANGUAGES.get(ext, 'text'), 'title': name}]
    return {'name': name, 'raw': raw, 'text': text, 'hash': digest(raw), 'ext': ext, 'records': records, 'oversized': ext not in {'.json', '.jsonl'} and len(raw) > 2 * 1024 * 1024}


class BodyParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden, self.title, self.in_title = [], 0, [], False

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'nav', 'footer', 'header', 'noscript'}:
            self.hidden += 1
        if tag == 'title':
            self.in_title = True
        if tag in {'p', 'div', 'li', 'br', 'pre', 'h1', 'h2', 'h3', 'h4', 'section'} and not self.hidden:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'nav', 'footer', 'header', 'noscript'}:
            self.hidden = max(0, self.hidden - 1)
        if tag == 'title':
            self.in_title = False
        if tag in {'p', 'div', 'li', 'pre', 'h1', 'h2', 'h3', 'h4', 'section'} and not self.hidden:
            self.parts.append('\n')

    def handle_data(self, value):
        if self.in_title:
            self.title.append(value)
        if not self.hidden:
            self.parts.append(value)


def normalize(text: str, language: str) -> tuple[str, str | None]:
    if language == 'html':
        parser = BodyParser()
        parser.feed(text)
        return re.sub(r'\n[ \t]*\n+', '\n\n', ''.join(parser.parts)).strip(), ''.join(parser.title).strip() or None
    return text.replace('\r\n', '\n').replace('\r', '\n'), None


def quality(text: str, oversized=False) -> tuple[str, list[str]]:
    reasons = []
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return 'quarantined', ['SECRET_DETECTED']
    if oversized:
        reasons.append('PARSER_SIZE_LIMIT')
    if INJECTION.search(text):
        reasons.append('INSTRUCTION_CONTAMINATION')
    if re.search(r'(?i)\b(?:social security|주민등록번호)\b|\b\d{6}-[1-4]\d{6}\b', text):
        reasons.append('PERSONAL_DATA_SUSPECTED')
    if any(len(line) > 4000 for line in text.splitlines()):
        reasons.append('MINIFIED_OR_LONG_LINE')
    if not text.strip():
        reasons.append('EMPTY_NORMALIZED_DOCUMENT')
    return ('needs_review' if reasons else 'passed'), reasons


def search_terms(text: str) -> str:
    pieces = re.findall(r'[\w./-]+', text, flags=re.UNICODE)
    expanded = []
    for piece in pieces:
        expanded.append(piece)
        expanded.extend(re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', piece).replace('_', ' ').replace('/', ' ').replace('.', ' ').replace('-', ' ').split())
    return ' '.join(expanded)


def text_language(text: str) -> str:
    ko = bool(re.search(r'[가-힣]', text))
    en = bool(re.search(r'[A-Za-z]', text))
    return 'mixed' if ko and en else 'ko' if ko else 'en' if en else 'unknown'


def lexical_units(text: str) -> list[tuple[int, int]]:
    # An explicitly labelled lexical estimate, never reported as model tokens.
    return [(m.start(), m.end()) for m in re.finditer(r'[A-Za-z0-9_]+|[^\s]', text)]


def make_chunks(text: str, language: str, counter=None, analysis=None) -> list[dict]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    boundaries = {0, len(lines)}
    symbols = {}
    if analysis:
        for item in analysis.get('symbols',[]):
            boundaries.update((max(0,item['startLine']-1),min(len(lines),item['endLine'])))
            symbols[item['startLine']-1]=item['name']
    fallback = False
    if language == 'python':
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) - 1
                    boundaries.update((start, node.end_lineno))
                    symbols[start] = node.name
        except (SyntaxError, ValueError, MemoryError):
            fallback = True
    elif language in {'javascript', 'typescript', 'sql'}:
        for i, line in enumerate(lines):
            match = re.search(r'(?:function|class|interface|type|const|let|export\s+default\s+function)\s+([A-Za-z_$][\w$]*)|(?:CREATE\s+(?:TABLE|VIEW|FUNCTION)\s+)(\w+)', line, re.I)
            if match:
                boundaries.add(i)
                symbols[i] = match.group(1) or match.group(2)
        fallback = len(boundaries) == 2
    else:
        in_fence = False
        for i, line in enumerate(lines):
            if line.lstrip().startswith(('```', '~~~')):
                in_fence = not in_fence
            if not in_fence and (line.startswith('#') or not line.strip()):
                boundaries.add(i)
    result = []
    ordered = sorted(boundaries)
    for a, b in zip(ordered, ordered[1:]):
        if a >= b:
            continue
        section = ''.join(lines[a:b])
        if not section.strip():
            continue
        symbol = symbols.get(a, '')
        chunks = [(0, len(section))]
        if (counter(section) if counter else len(lexical_units(section))) > (256 if fallback else 448):
            units = lexical_units(section)
            chunks = []
            cursor = 0
            while cursor < len(units):
                end_index = min(cursor + (256 if fallback else 300), len(units))
                end = units[end_index - 1][1]
                begin = units[cursor][0]
                # Enforce the actual embedding tokenizer budget when installed.
                if counter:
                    while end_index > cursor + 1 and counter(section[begin:end]) > 448:
                        end_index -= 1
                        end = units[end_index - 1][1]
                chunks.append((begin, end))
                if end_index == len(units):
                    break
                cursor = max(cursor + 1, end_index - 32)
        for begin, end in chunks:
            content = section[begin:end]
            raw_start, raw_end = starts[a] + begin, starts[a] + end
            start_line = text.count('\n', 0, raw_start) + 1
            end_line = text.count('\n', 0, max(raw_start, raw_end - 1)) + 1
            result.append({'text': content, 'symbol': symbol, 'section': next((line.strip('# \n\r') for line in lines[:a + 1][::-1] if line.startswith('#')), ''), 'startLine': start_line, 'endLine': end_line, 'startOffset': raw_start, 'endOffset': raw_end, 'parserFallback': fallback, 'embeddingTokenCount': counter(content) if counter else None, 'lexicalUnitEstimate': len(lexical_units(content))})
    return result
