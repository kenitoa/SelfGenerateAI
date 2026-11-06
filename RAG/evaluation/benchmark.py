"""Small authored coding corpus, bilingual development questions and no-answer cases.

This is a regression benchmark, not a claim about internet-wide code quality.
"""
VERSION='seed-code-dev-1'
CORPUS={
 'python/unique.py':'''def stable_unique(values):
    """Remove duplicate items while preserving the original input order."""
    return list(dict.fromkeys(values))
''',
 'python/retry.py':'''import time

def retry_operation(operation, attempts=3, delay=0.1):
    """Retry transient failures with an exponential backoff delay."""
    for attempt in range(attempts):
        try:
            return operation()
        except TimeoutError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * 2 ** attempt)
''',
 'python/paths.py':'''from pathlib import Path

def safe_child(root, relative):
    """Reject directory traversal outside the configured workspace."""
    base = Path(root).resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("outside workspace")
    return target
''',
 'python/pagination.py':'''def paginate(items, page, page_size):
    """Return a bounded page of results from a zero-based page number."""
    if page < 0 or page_size < 1:
        raise ValueError("invalid pagination")
    start = page * page_size
    return items[start:start + page_size]
''',
 'typescript/debounce.ts':'''export function debounce(callback: (...args: unknown[]) => void, delay: number) {
  // Delay execution until calls stop arriving for the timeout interval.
  let timer: ReturnType<typeof setTimeout>;
  return (...args: unknown[]) => {
    clearTimeout(timer);
    timer = setTimeout(() => callback(...args), delay);
  };
}
''',
 'javascript/group.js':'''export function groupBy(items, key) {
  // Group records into buckets using a selected property.
  return items.reduce((groups, item) => {
    const value = item[key];
    (groups[value] ??= []).push(item);
    return groups;
  }, {});
}
''',
 'go/cancel.go':'''package worker
import "context"
func WaitForValue(ctx context.Context, values <-chan string) (string, error) {
    // Stop waiting immediately when the context is cancelled.
    select {
    case <-ctx.Done(): return "", ctx.Err()
    case value := <-values: return value, nil
    }
}
''',
 'rust/parse.rs':'''pub fn parse_port(value: &str) -> Result<u16, String> {
    // Parse a network port without panicking on invalid input.
    value.parse::<u16>().map_err(|_| "invalid port".to_string())
}
''',
 'java/LruCache.java':'''import java.util.LinkedHashMap;
import java.util.Map;
public class LruCache<K,V> extends LinkedHashMap<K,V> {
    private final int capacity;
    public LruCache(int capacity) { super(16, 0.75f, true); this.capacity = capacity; }
    // Evict the least recently used entry once the size exceeds capacity.
    protected boolean removeEldestEntry(Map.Entry<K,V> eldest) { return size() > capacity; }
}
''',
 'cpp/search.cpp':'''#include <vector>
int binary_search_index(const std::vector<int>& values, int target) {
    // Find an element in a sorted array using logarithmic search.
    int low = 0, high = static_cast<int>(values.size()) - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (values[mid] == target) return mid;
        if (values[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}
''',
}

TOPICS=[
 ('python/unique.py','stable_unique','중복을 제거하면서 입력 순서를 유지하는 코드','Remove duplicates and keep the original order'),
 ('python/retry.py','retry_operation','시간 초과 실패 시 지수 백오프로 재시도하는 함수','Retry a failed operation with exponential backoff'),
 ('python/paths.py','safe_child','작업 폴더 밖으로 경로가 벗어나는 것을 차단하는 방법','Reject a path that escapes the workspace directory'),
 ('python/pagination.py','paginate','목록을 페이지 크기로 나누어 일부만 가져오기','Get a page of results from a list'),
 ('typescript/debounce.ts','debounce','입력이 멈춘 뒤에만 콜백을 실행하는 함수','Delay a callback until repeated calls stop'),
 ('javascript/group.js','groupBy','배열의 객체를 특정 속성별로 묶는 함수','Group objects by a selected property'),
 ('go/cancel.go','WaitForValue','취소 신호가 오면 채널 대기를 끝내는 코드','Cancel waiting on a channel when a context ends'),
 ('rust/parse.rs','parse_port','잘못된 포트 번호를 패닉 없이 오류로 반환하기','Parse a network port without a panic'),
 ('java/LruCache.java','LruCache','오래 사용하지 않은 항목을 용량 초과 시 제거하는 캐시','Evict least recently used entries when a cache is full'),
 ('cpp/search.cpp','binary_search_index','정렬된 배열에서 이진 탐색으로 위치 찾기','Find an index in a sorted array with binary search'),
]

def questions():
    rows=[]
    for index,(path,symbol,ko,en) in enumerate(TOPICS):
        for language,query in [('ko',ko),('en',en)]:
            rows.append({'id':f'{index}-{language}','query':query,'expectedPaths':[path],'expectedSymbol':symbol,'questionLanguage':language,'case':'natural_language'})
        rows.append({'id':f'{index}-symbol','query':symbol,'expectedPaths':[path],'expectedSymbol':symbol,'questionLanguage':'symbol','case':'symbol'})
    for index,query in enumerate(['missing_cryptographic_wallet_signer','nonexistent_kubernetes_scheduler','없는_양자회로_컴파일러','undefined_satellite_orbit_predictor','crochet sweater knitting pattern','초콜릿 케이크를 굽는 온도']):
        rows.append({'id':f'negative-{index}','query':query,'expectedPaths':[],'questionLanguage':'ko' if index in (2,5) else 'en','case':'no_answer'})
    return rows
