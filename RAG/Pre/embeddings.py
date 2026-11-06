"""Optional local-only E5 ONNX encoder. No downloads or remote model code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LocalEncoder:
    def __init__(self, config):
        self.reason = 'EMBEDDING_NOT_CONFIGURED'
        self.session = self.tokenizer = self.profile = None
        self._config = config or {}
        if self._config.get('enabled') is False:
            return
        if not self._config.get('directory') and not self._config.get('modelPath'):
            return
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
            directory = Path(self._config.get('directory') or Path(self._config['modelPath']).parent).expanduser().resolve()
            manifest = json.loads((directory / 'profile.json').read_text(encoding='utf-8'))
            model = Path(self._config['modelPath']).expanduser().resolve() if self._config.get('modelPath') else directory / 'model.onnx'
            tokenizer = Path(self._config['tokenizerPath']).expanduser().resolve() if self._config.get('tokenizerPath') else directory / 'tokenizer.json'
            for item in (model, tokenizer):
                expected = manifest.get('checksums', {}).get(item.name)
                hashed=hashlib.sha256()
                with item.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024**2),b''):hashed.update(block)
                actual = hashed.hexdigest()
                if not expected or expected != actual:
                    raise ValueError('model/tokenizer checksum mismatch')
            if manifest.get('dimension') != 384 or not manifest.get('modelRevision') or not manifest.get('tokenizerRevision'):
                raise ValueError('profile revision/dimension missing')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
            self.session = ort.InferenceSession(str(model), sess_options=opts, providers=['CPUExecutionProvider'])
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
            self.tokenizer.enable_padding()
            self.tokenizer.no_truncation()
            self.profile = {**manifest, 'id': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:32], 'pooling': 'attention_mask_mean', 'normalization': 'l2'}
            self.reason = None
        except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
            self.reason = f'EMBEDDING_UNAVAILABLE:{type(exc).__name__}'
            self.session = self.tokenizer = self.profile = None

    @property
    def available(self):
        return self.session is not None

    def count(self, text):
        if not self.available:
            return None
        return len(self.tokenizer.encode('passage: ' + text).ids)

    def encode(self, values, query=False):
        import numpy as np
        if not self.available:
            raise RuntimeError(self.reason)
        output = []
        for value in values:
            encoded = self.tokenizer.encode(('query: ' if query else 'passage: ') + value)
            if len(encoded.ids) > 512:
                raise ValueError('Embedding input exceeds 512 tokens')
            inputs = {'input_ids': np.array([encoded.ids], dtype=np.int64), 'attention_mask': np.array([encoded.attention_mask], dtype=np.int64), 'token_type_ids': np.array([encoded.type_ids], dtype=np.int64)}
            inputs = {name: inputs[name] for name in (item.name for item in self.session.get_inputs())}
            hidden = self.session.run(None, inputs)[0]
            if hidden.ndim == 3:
                mask = inputs['attention_mask'][..., None]
                vector = (hidden * mask).sum(axis=1)[0] / max(float(mask.sum()), 1.0)
            else:
                vector = hidden[0]
            vector = vector.astype(np.float32)
            if vector.shape != (384,) or not np.isfinite(vector).all():
                raise ValueError('Invalid embedding output')
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            output.append(vector)
        return np.stack(output) if output else np.empty((0, 384), dtype=np.float32)
