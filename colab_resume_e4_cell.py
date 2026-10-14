# E4_ARTIFACT_RECOVERY: ONE SELF-CONTAINED COLAB CELL
# Chèn sau E4 cũ, trước bảng kết quả; chạy cell này và các cell bên dưới.
# Không cần chạy cell phía trên hoặc có bản notebook gốc trên Drive.

from pathlib import Path
import os
import sys
import json
import copy

# Thư mục artifact cũ đã truy cập được từ Drive mount (không phải URL chia sẻ).
SOURCE_ROOT = Path('/content/drive/MyDrive/lab_day10_colab_runs/run_all_v1')
# Thư mục ghi E4 và báo cáo mới. Mặc định giữ riêng với artifact E0–E3.
SAVED_ROOT = Path('/content/drive/MyDrive/lab_day10_colab_runs/run_all_v1_e4')
CREDENTIALS = {'KAGGLE_API_TOKEN': '', 'KAGGLE_USERNAME': '', 'KAGGLE_KEY': '', 'HF_TOKEN': ''}
# Có thể để trống CREDENTIALS nếu đã bật Notebook access trong Colab Secrets.
INFER_BATCH_SIZE = 4
E4_BASE_SETUP = 'auto'
MAKE_FIGURES = True
DATASET_SOURCE = None
DATASET_ZIP = None

RUN_MODE = 'full'
RUNS = ('E4',)
RUN_LABEL = 'run_all_v1_e4'
SAVE_TO_DRIVE = True
RESTORE_FROM_DRIVE = True
RESTART_INCOMPLETE = True
OUTPUT_ROOT = Path('/content/lab_day10_e4_outputs') / RUN_LABEL
REPO_ROOT = Path('/content/lab_day10_repo')
TEMP_ROOT = Path('/content/lab_day10_cache')
REPO_URL = 'https://github.com/nthanhwork/K4-Track-4-Day10-VisionLanguageModel.git'
assert INFER_BATCH_SIZE > 0
assert E4_BASE_SETUP in ('auto', 'E1', 'E2', 'E3')
assert OUTPUT_ROOT.resolve() not in (SOURCE_ROOT.resolve(), SAVED_ROOT.resolve())

from google.colab import drive
drive.mount('/content/drive', force_remount=False)
if not SOURCE_ROOT.is_dir():
    raise FileNotFoundError(f'Không thấy thư mục artifact {SOURCE_ROOT}. Sửa SOURCE_ROOT.')

BASELINE_RUNS = {}
for _name in ('E1', 'E2', 'E3'):
    _final = SOURCE_ROOT / 'work/full' / _name.lower() / 'final'
    _record = _final / 'run.json'
    if not _record.is_file():
        raise FileNotFoundError(f'Thiếu checkpoint final {_record}; không tự chạy lại E1–E3.')
    _run = json.loads(_record.read_text())
    if not _run.get('complete') or _run.get('completed_steps') != 300:
        raise RuntimeError(f'{_name}: checkpoint chưa đủ 300 bước.')
    if not (_final / 'adapter/adapter_config.json').is_file() or not any(
            p.stat().st_size > 0 for p in (_final / 'adapter').glob('*.safetensors')):
        raise RuntimeError(f'{_name}: thiếu adapter trên Drive.')
    if _run['config']['train_projector'] and not (_final / 'extra_trainable.pt').is_file():
        raise RuntimeError(f'{_name}: thiếu projector trên Drive.')
    if (_run['config']['name'] != _name or _run['config']['setup'] != _name
            or _run['config']['max_steps'] != 300 or _run['config']['train_fraction'] != 1.0):
        raise RuntimeError(f'{_name}: cấu hình baseline không khớp setup 100% dữ liệu.')
    BASELINE_RUNS[_name] = _run
for _key in ('model_id', 'model_revision', 'dataset_version', 'prompt'):
    if len({r[_key] for r in BASELINE_RUNS.values()}) != 1:
        raise RuntimeError(f'E1–E3 khác {_key}; kiểm tra artifact trước khi chạy E4.')
MODEL_ID = BASELINE_RUNS['E1']['model_id']
MODEL_REVISION = BASELINE_RUNS['E1']['model_revision']
PROMPT_SAVED = BASELINE_RUNS['E1']['prompt']
TRAIN_BATCH_SIZE = BASELINE_RUNS['E1']['config']['batch_size']
GRAD_ACCUM = BASELINE_RUNS['E1']['config']['grad_accum']

_val_manifests = []
_test_manifests = []
for _name in BASELINE_RUNS:
    _base = SOURCE_ROOT / 'predictions/full'
    _val_manifests.append(json.loads((_base / f'{_name.lower()}_val.manifest.json').read_text()))
    _test_manifests.append(json.loads((_base / f'{_name.lower()}_test.manifest.json').read_text()))
if len({m['max_new_tokens'] for m in _val_manifests + _test_manifests}) != 1:
    raise RuntimeError('MAX_NEW_TOKENS giữa các baseline khác nhau.')
if len({m['n_images'] for m in _val_manifests}) != 1:
    raise RuntimeError('Các baseline không dùng cùng số ảnh val.')
MAX_NEW_TOKENS = _val_manifests[0]['max_new_tokens']
VAL_LIMIT = _val_manifests[0]['n_images']
assert MAX_NEW_TOKENS > 0 and VAL_LIMIT > 0

for directory in (OUTPUT_ROOT, TEMP_ROOT, SAVED_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(TEMP_ROOT / 'huggingface')
os.environ['HF_HUB_CACHE'] = str(TEMP_ROOT / 'huggingface/hub')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print('Nguồn E0–E3:', SOURCE_ROOT)
print('Lưu E4/báo cáo:', SAVED_ROOT)
print('Model commit:', MODEL_REVISION, '| val:', VAL_LIMIT, '| max tokens:', MAX_NEW_TOKENS)


from google.colab import userdata, drive


def read_credential(name):
    # Ưu tiên giá trị đã điền, rồi biến môi trường, rồi Colab Secrets.
    value = CREDENTIALS.get(name) or os.environ.get(name)
    if not value:
        try:
            value = userdata.get(name)
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
            value = ''
    return str(value).strip() if value else ''


def configure_credentials():
    values = {name: read_credential(name) for name in CREDENTIALS}
    if DATASET_SOURCE is None and DATASET_ZIP is None:
        if not values['KAGGLE_API_TOKEN'] and not (
                values['KAGGLE_USERNAME'] and values['KAGGLE_KEY']):
            raise RuntimeError(
                'Thiếu credential Kaggle. Điền KAGGLE_API_TOKEN hoặc '
                'KAGGLE_USERNAME + KAGGLE_KEY ở cell đầu / Colab Secrets '
                '(bật Notebook access), rồi Run all.')
    for name, value in values.items():
        if value:
            os.environ[name] = value
    # Chỉ in trạng thái, không in giá trị credential.
    print('Credential:', 'API token' if values['KAGGLE_API_TOKEN'] else
          'legacy username/key' if values['KAGGLE_USERNAME'] and values['KAGGLE_KEY'] else
          'không cần cho dataset local')


configure_credentials()
if SAVE_TO_DRIVE or RESTORE_FROM_DRIVE:
    drive.mount('/content/drive', force_remount=False)
    assert Path('/content/drive/MyDrive').is_dir(), 'Cần kết nối Drive để lưu checkpoint.'
    SAVED_ROOT.mkdir(parents=True, exist_ok=True)
print('Chạy:', RUN_MODE, RUNS, '| kết quả:', OUTPUT_ROOT)


import importlib.metadata
import subprocess

if not (REPO_ROOT / 'scripts/score.py').exists():
    if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
        raise RuntimeError(f'{REPO_ROOT} có dữ liệu nhưng không phải repo lab.')
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)], check=True)
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PINNED = ['transformers==5.19.0', 'peft==0.21.2', 'trl==1.9.2',
          'accelerate==1.15.0', 'bitsandbytes==0.50.2']
EXTRAS = ['pyyaml', 'pillow', 'kaggle', 'pytest', 'matplotlib']
for package in PINNED:
    name, wanted = package.split('==')
    if name in sys.modules and getattr(sys.modules[name], '__version__', None) != wanted:
        raise RuntimeError(f'{name} đã import bản khác. Restart session rồi Run All.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINNED, *EXTRAS], check=True)
# Cập nhật CLI để dùng được cả API token mới và credential legacy.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'kaggle'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
if any(name == 'torchao' or name.startswith('torchao.') for name in sys.modules):
    raise RuntimeError('torchao đã được import trước khi gỡ. Restart session rồi Run All.')
subprocess.run(['nvidia-smi', '-L'], check=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=REPO_ROOT, check=True)


import copy
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
import uuid
from collections import Counter

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from IPython.display import Markdown, display
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import evaluate, parse_output, iou_xyxy
from peft.import_utils import is_torchao_available

assert torch.cuda.is_available(), 'Chọn Runtime → Change runtime type → T4 GPU.'
assert is_torchao_available() is False, 'Restart session sau khi gỡ torchao.'
DEVICE = torch.device('cuda:0')
DATA_ROOT = TEMP_ROOT / 'data'
CACHE_ROOT = TEMP_ROOT / 'huggingface/hub'
WORK_ROOT = OUTPUT_ROOT / 'work'
PREDICTION_ROOT = OUTPUT_ROOT / 'predictions'
REPORT_ROOT = OUTPUT_ROOT / 'report'
EVIDENCE_ROOT = REPORT_ROOT / 'evidence'
for directory in (WORK_ROOT, PREDICTION_ROOT, EVIDENCE_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


def backup_path(source):
    """Sao lưu file/thư mục kết quả; file marker ghi cuối, không chứa credential."""
    if not SAVE_TO_DRIVE:
        return
    source = Path(source)
    relative = source.resolve().relative_to(OUTPUT_ROOT.resolve())
    destination = SAVED_ROOT / relative
    if source.resolve() == destination.resolve():
        return
    files = [source] if source.is_file() else [p for p in source.rglob('*') if p.is_file()]
    files.sort(key=lambda p: (p.name.endswith(('run.json', '.manifest.json')), str(p)))
    for file in files:
        if file.name.endswith(('.tmp', '.partial', '.uploading')):
            continue
        dest = destination if source.is_file() else destination / file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == file.stat().st_size and (
                dest.stat().st_mtime_ns == file.stat().st_mtime_ns):
            continue
        temporary = dest.with_name(dest.name + '.uploading')
        shutil.copy2(file, temporary)
        temporary.replace(dest)


def restore_outputs():
    if not RESTORE_FROM_DRIVE or not SAVED_ROOT.exists():
        return
    files = [p for p in SAVED_ROOT.rglob('*') if p.is_file()]
    files.sort(key=lambda p: (p.name.endswith(('run.json', '.manifest.json')), str(p)))
    restored = 0
    for source in files:
        if source.name.endswith(('.tmp', '.partial', '.uploading')):
            continue
        destination = OUTPUT_ROOT / source.relative_to(SAVED_ROOT)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            restored += 1
    print('Khôi phục từ Drive:', restored, 'file')



def _restore_artifacts(source_root, skip_work=False):
    source_root = Path(source_root)
    if not source_root.exists():
        return
    files = [p for p in source_root.rglob('*') if p.is_file()]
    files.sort(key=lambda p: (p.name.endswith(('run.json', '.manifest.json')), str(p)))
    restored = 0
    for file in files:
        relative = file.relative_to(source_root)
        if skip_work and relative.parts[0] == 'work':
            continue
        if file.name.endswith(('.tmp', '.partial', '.uploading')):
            continue
        destination = OUTPUT_ROOT / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
            restored += 1
    print('Khôi phục từ', source_root, ':', restored, 'file')

# Baseline checkpoint đọc trực tiếp ở SOURCE_ROOT, không sao chép GB trọng số.
_restore_artifacts(SOURCE_ROOT, skip_work=True)
_restore_artifacts(SAVED_ROOT)
if not (OUTPUT_ROOT / 'REPORT.md').exists():
    shutil.copy2(REPO_ROOT / 'REPORT.md', OUTPUT_ROOT / 'REPORT.md')
backup_path(OUTPUT_ROOT / 'REPORT.md')
print('GPU:', torch.cuda.get_device_name(0))


command = [sys.executable, str(REPO_ROOT / 'scripts/prepare_data.py'),
           '--out', str(DATA_ROOT), '--work', str(TEMP_ROOT / 'prepare')]
if DATASET_ZIP is not None:
    command += ['--zip', str(DATASET_ZIP)]
elif DATASET_SOURCE is not None:
    command += ['--source', str(DATASET_SOURCE)]
else:
    command += ['--kaggle']
subprocess.run(command, check=True)


splits = {
    split: read_jsonl(DATA_ROOT / 'splits' / f'{split}.jsonl')
    for split in ('train', 'val', 'test')
}
train_samples, val_samples, test_samples = (splits[s] for s in ('train', 'val', 'test'))
dataset_meta = json.loads((DATA_ROOT / 'splits/meta.json').read_text())
DATASET_VERSION = dataset_meta['dataset_version']

session_sets = {s: {e['session'] for e in rows} for s, rows in splits.items()}
for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    assert session_sets[a].isdisjoint(session_sets[b]), f'Rò rỉ session: {a}/{b}'
for split, rows in splits.items():
    assert rows and len({e['image_id'] for e in rows}) == len(rows)
    counts = Counter(o['label'] for e in rows for o in e['objects'])
    print(split, 'ảnh:', len(rows), 'box:', dict(counts), 'sessions:', len(session_sets[split]))
print('dataset_version:', DATASET_VERSION)


def load_image(entry):
    # image_path tương đối với data/, không phải thư mục notebook.
    with Image.open(DATA_ROOT / entry['image_path']) as img:
        image = img.convert('RGB')
    assert image.size == (entry['width'], entry['height'])
    return image


def draw_boxes(ax, boxes, width, height, color, coord='norm1000', linestyle='-'):
    from matplotlib.patches import Rectangle
    sx, sy = (width / 1000, height / 1000) if coord == 'norm1000' else (1, 1)
    for box in boxes:
        x1, y1, x2, y2 = box.get('bbox_2d', box.get('bbox'))
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=1.5,
                               linestyle=linestyle))
        ax.text(x1, y1, box['label'], color=color, fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))




import math

TARGET_POLICY = 'skip_zero_area_round_tiny_outward_v1'


def yolo_to_box(label, yolo):
    assert label in CLASSES and len(yolo) == 4, (label, yolo)
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in yolo), (label, yolo)
    xc, yc, w, h = yolo
    # Nhãn nguồn có w/h = 0: bỏ khỏi target, giữ nguyên split và GT chấm điểm.
    if w == 0 or h == 0:
        return None
    coordinates = [(xc - w / 2) * 1000, (yc - h / 2) * 1000,
                   (xc + w / 2) * 1000, (yc + h / 2) * 1000]
    coordinates = [max(0.0, min(1000.0, v)) for v in coordinates]
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        return None
    box = [round(v) for v in coordinates]
    # Box rất nhỏ có diện tích dương có thể bị round thành một đường.
    for lo, hi in ((0, 2), (1, 3)):
        if box[hi] <= box[lo]:
            box[lo] = math.floor(coordinates[lo])
            box[hi] = math.ceil(coordinates[hi])
    assert box[2] > box[0] and box[3] > box[1], (label, yolo)
    return {'label': label, 'bbox_2d': box}


def objects_to_answer(objects):
    boxes = []
    for obj in objects:
        box = yolo_to_box(obj['label'], obj['yolo'])
        if box is not None:
            boxes.append(box)
    boxes.sort(key=lambda b: (b['bbox_2d'][1], b['bbox_2d'][0]))
    return json.dumps(boxes, separators=(',', ':'))


PROMPT = (
    'Locate all potholes, cracks, and manholes in this image. '
    'Return only a JSON list of objects with keys label and bbox_2d. '
    'Labels must be pothole, crack, or manhole. '
    'bbox_2d is [x1,y1,x2,y2] using integer coordinates from 0 to 1000 '
    'relative to image width and height. Return [] if no objects are present.'
)


PROMPT = PROMPT_SAVED
FINGERPRINT_SCHEMA = None


from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, set_seed
from bitsandbytes.nn import Linear4bit
model_config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE_ROOT)
MODEL_COMMIT = getattr(model_config, '_commit_hash', None) or MODEL_REVISION
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_COMMIT, cache_dir=CACHE_ROOT)
processor.tokenizer.padding_side = 'left'
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
print('Model commit:', MODEL_COMMIT)

def load_base_model():
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['model.visual', 'lm_head'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_COMMIT, cache_dir=CACHE_ROOT, quantization_config=quantization, dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa')
    bad = [n for n, m in model.model.visual.named_modules() if isinstance(m, Linear4bit)]
    assert not bad, f'ViT bị lượng tử hóa: {bad[:5]}'
    assert not isinstance(model.lm_head, Linear4bit)
    assert any((isinstance(m, Linear4bit) for m in model.model.language_model.modules()))
    return model

def user_message(image, prompt=PROMPT):
    return {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}

def to_device(batch):
    return {key: value.to(DEVICE) for key, value in batch.items()}

def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()

def config_fingerprint(config):
    payload = {'config': config, 'model': MODEL_ID, 'revision': MODEL_COMMIT, 'dataset': DATASET_VERSION, 'prompt': PROMPT, 'mode': RUN_MODE, 'packages': PINNED}
    if FINGERPRINT_SCHEMA == 'target_policy_v1':
        payload['target_policy'] = TARGET_POLICY
    elif FINGERPRINT_SCHEMA != 'legacy':
        raise RuntimeError('Chưa xác minh schema fingerprint từ artifact.')
    return fingerprint(payload)

def run_directory(name):
    if name.upper() in {'E0', 'E1', 'E2', 'E3'}:
        return SOURCE_ROOT / 'work/full' / name.lower()
    return WORK_ROOT / 'full' / name.lower()

def prediction_path(name, split):
    folder = PREDICTION_ROOT / RUN_MODE
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{name.lower()}_{split}.jsonl'

def prediction_manifest(run, entries, split):
    return {'run_id': run['run_id'], 'fingerprint': run['fingerprint'], 'split': split, 'max_new_tokens': MAX_NEW_TOKENS, 'image_ids_sha256': fingerprint([e['image_id'] for e in entries]), 'n_images': len(entries)}

def valid_prediction_file(path, entries, expected):
    path = Path(path)
    try:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        if any((manifest.get(k) != value for k, value in expected.items())):
            return False
        rows = read_jsonl(path)
        ids = [row['image_id'] for row in rows]
        return len(ids) == len(entries) and len(set(ids)) == len(ids) and (set(ids) == {e['image_id'] for e in entries}) and all((isinstance(row.get('raw'), str) for row in rows))
    except (OSError, ValueError, KeyError, TypeError):
        return False

@torch.inference_mode()
def predict_jsonl(model, entries, out_path, manifest):
    model.eval()
    model.gradient_checkpointing_disable()
    processor.tokenizer.padding_side = 'left'
    out_path = Path(out_path)
    partial = out_path.with_suffix('.partial')
    with partial.open('w', encoding='utf-8') as stream:
        for start in range(0, len(entries), INFER_BATCH_SIZE):
            rows = entries[start:start + INFER_BATCH_SIZE]
            messages = [[user_message(load_image(e))] for e in rows]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', padding=True)
            inputs.pop('token_type_ids', None)
            inputs = to_device(inputs)
            with torch.autocast('cuda', dtype=torch.float16):
                generated = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.pad_token_id)
            raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for entry, text in zip(rows, raw):
                stream.write(json.dumps({'image_id': entry['image_id'], 'raw': text}, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"Inference {manifest['split']}: {min(start + INFER_BATCH_SIZE, len(entries))}/{len(entries)}")
            del inputs, generated
    partial.replace(out_path)
    out_path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    assert valid_prediction_file(out_path, entries, manifest)
    backup_path(out_path)
    backup_path(out_path.with_suffix('.manifest.json'))

def score_predictions(path, name, run):
    assert RUN_MODE == 'full', 'Smoke không được tạo evidence test.'
    assert valid_prediction_file(path, test_samples, prediction_manifest(run, test_samples, 'test'))
    out = EVIDENCE_ROOT / f'{name.lower()}.json'
    meta = {key: run[key] for key in ('trainable_params', 'peak_vram_gb', 'train_minutes')}
    meta.update(run_id=run['run_id'], fingerprint=run['fingerprint'], base_setup=run['config']['setup'], train_fraction=run['config']['train_fraction'])
    if run['config']['setup'] == 'E0':
        meta['inference_minutes'] = run.get('inference_minutes', 0)
    command = [sys.executable, str(REPO_ROOT / 'scripts/score.py'), '--pred', str(path), '--splits-dir', str(DATA_ROOT / 'splits'), '--split', 'test', '--name', name, '--out', str(out)]
    for key, value in meta.items():
        command += ['--meta', f'{key}={value}']
    subprocess.run(command, check=True)
    backup_path(out)
    return json.loads(out.read_text())


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def encode_example(image, prompt, answer, processor):
    user = user_message(image, prompt)
    assistant = {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}
    prompt_enc = processor.apply_chat_template(
        [user], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt', padding=False)
    full_enc = processor.apply_chat_template(
        [user, assistant], tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt', padding=False)
    n_prompt = prompt_enc['input_ids'].shape[1]
    assert torch.equal(full_enc['input_ids'][0, :n_prompt], prompt_enc['input_ids'][0]),         'Prompt không phải tiền tố: kiểm tra chat template.'
    sample = {k: full_enc[k][0] for k in ('input_ids', 'attention_mask')}
    if 'mm_token_type_ids' in full_enc:
        sample['mm_token_type_ids'] = full_enc['mm_token_type_ids'][0]
    sample['labels'] = sample['input_ids'].clone()
    sample['labels'][:n_prompt] = -100
    sample['pixel_values'] = full_enc['pixel_values']
    sample['image_grid_thw'] = full_enc['image_grid_thw']
    assert (sample['labels'] != -100).any(), 'Không có token đáp án.'
    return sample


def collate_fn(batch, pad_token_id=None):
    pad_token_id = processor.tokenizer.pad_token_id if pad_token_id is None else pad_token_id
    result = {
        'input_ids': pad_sequence([b['input_ids'] for b in batch], batch_first=True,
                                  padding_value=pad_token_id),
        'attention_mask': pad_sequence([b['attention_mask'] for b in batch],
                                       batch_first=True, padding_value=0),
        'labels': pad_sequence([b['labels'] for b in batch], batch_first=True,
                               padding_value=-100),
        'pixel_values': torch.cat([b['pixel_values'] for b in batch], dim=0),
        'image_grid_thw': torch.cat([b['image_grid_thw'] for b in batch], dim=0),
    }
    if 'mm_token_type_ids' in batch[0]:
        assert all('mm_token_type_ids' in b for b in batch)
        result['mm_token_type_ids'] = pad_sequence([b['mm_token_type_ids'] for b in batch],
                                                  batch_first=True, padding_value=0)
    assert (result['labels'][result['attention_mask'] == 0] == -100).all()
    return result


class RoadDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        return encode_example(load_image(entry), PROMPT,
                              objects_to_answer(entry['objects']), processor)


# Kiểm tra loss mask và số token ảnh mà không cần nạp model.
sample = RoadDataset(train_samples)[0]
image_token_count = int((sample['input_ids'] == processor.image_token_id).sum())
grid_count = int(sample['image_grid_thw'].prod(dim=-1).sum()) // processor.image_processor.merge_size**2
assert image_token_count == grid_count == 220, (image_token_count, grid_count)
assert (sample['labels'][sample['input_ids'] == processor.image_token_id] == -100).all()
decoded_answer = processor.tokenizer.decode(
    sample['labels'][sample['labels'] != -100], skip_special_tokens=False)
assert decoded_answer.startswith(objects_to_answer(train_samples[0]['objects']))
assert '<|im_end|>' in decoded_answer
print('Token ảnh:', image_token_count, '\nToken được tính loss:', decoded_answer)
batch_check = collate_fn([sample, RoadDataset(train_samples)[1]])
print({k: tuple(v.shape) for k, v in batch_check.items()})
del sample, batch_check


from peft import LoraConfig, PeftModel, get_peft_model

def is_projector(name):
    return '.visual.merger.' in name or '.visual.deepstack_merger_list.' in name

def lora_pattern(setup_name):
    assert setup_name in ('E1', 'E2', 'E3')
    llm = 'model\\.language_model\\.layers\\.\\d+\\.(?:self_attn\\.(?:q|k|v|o)_proj|mlp\\.(?:gate|up|down)_proj)'
    vit = 'model\\.visual\\.blocks\\.\\d+\\.(?:attn\\.(?:qkv|proj)|mlp\\.(?:linear_fc1|linear_fc2))'
    return f'(?:{llm}|{vit})' if setup_name == 'E3' else llm

def load_config(name):
    name = name.upper()
    if name not in BASELINE_RUNS:
        raise ValueError(f'Chỉ đọc config baseline E1–E3; nhận {name}.')
    return copy.deepcopy(BASELINE_RUNS[name]['config'])

def apply_setup(model, setup_name, config=None):
    setup_name = setup_name.upper()
    config = config or load_config(setup_name)
    pattern = lora_pattern(setup_name)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    expected = 348 if setup_name == 'E3' else 252
    assert len(matched) == expected, f'Khớp {len(matched)}/{expected}: xem model.named_modules().'
    for param in model.parameters():
        param.requires_grad_(False)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=config['lora']['r'], lora_alpha=config['lora']['alpha'], lora_dropout=config['lora']['dropout'], target_modules=pattern, bias='none', task_type='CAUSAL_LM'))
    for name, param in model.named_parameters():
        if config['train_projector'] and is_projector(name):
            param.requires_grad_(True)
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            assert 'lora_' in name or (config['train_projector'] and is_projector(name))
    projector_n = sum((p.numel() for n, p in model.named_parameters() if p.requires_grad and is_projector(n)))
    if setup_name in ('E2', 'E3'):
        assert 100000000 < projector_n < 120000000, projector_n
    else:
        assert projector_n == 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.print_trainable_parameters()
    print('LoRA modules:', len(matched), 'Projector trainable:', projector_n)
    return model


from transformers import get_cosine_schedule_with_warmup


def select_training_entries(entries, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('train_fraction phải trong (0,1].')
    indices = random.Random(seed).sample(range(len(entries)), len(entries))
    return [entries[i] for i in indices[:max(1, round(len(entries) * fraction))]]


def load_completed_run(directory, config):
    directory = Path(directory)
    record_path = directory / 'run.json'
    if not record_path.exists():
        return None
    run = json.loads(record_path.read_text())
    if run.get('fingerprint') != config_fingerprint(config):
        raise RuntimeError(f'{directory}: config/model/dataset đã thay đổi. Dùng OUTPUT_ROOT mới.')
    if not run.get('complete') or run['completed_steps'] != config['max_steps']:
        return None
    if not (directory / 'adapter/adapter_config.json').exists():
        raise RuntimeError(f'{directory}: thiếu adapter_config.json.')
    if not any((directory / 'adapter').glob('*.safetensors')):
        raise RuntimeError(f'{directory}: thiếu trọng số adapter.')
    if config['train_projector'] and not (directory / 'extra_trainable.pt').exists():
        raise RuntimeError(f'{directory}: thiếu projector.')
    return run


def save_checkpoint(model, directory, run, history):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / 'adapter', save_embedding_layers=False)
    processor.save_pretrained(directory / 'processor')
    if run['config']['train_projector']:
        extra = {n: p.detach().float().cpu().contiguous()
                 for n, p in model.get_base_model().named_parameters() if is_projector(n)}
        assert extra
        temporary = directory / 'extra_trainable.pt.tmp'
        torch.save(extra, temporary)
        temporary.replace(directory / 'extra_trainable.pt')
    (directory / 'loss.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
    selection = directory.parent / 'train_image_ids.json'
    if selection.exists():
        shutil.copy2(selection, directory / selection.name)
    # Manifest ghi cuối, sau khi mọi trọng số đã được lưu.
    (directory / 'run.json').write_text(json.dumps(run, indent=2))
    backup_path(directory)
    print('Checkpoint đã lưu:', directory)


def train_experiment(config):
    config = copy.deepcopy(config)
    directory = run_directory(config['name'])
    final = directory / 'final'
    completed = load_completed_run(final, config)
    if completed:
        print('Dùng lại checkpoint hoàn tất:', final)
        backup_path(final)
        return final
    if directory.exists() and any(directory.iterdir()):
        if not RESTART_INCOMPLETE:
            raise RuntimeError(f'{directory} dở dang. Bật RESTART_INCOMPLETE để train lại từ 0.')
        print('Run dở dang: bắt đầu lại từ bước 0; không resume optimizer.')
        for record in directory.glob('step-*/run.json'):
            if json.loads(record.read_text()).get('fingerprint') != config_fingerprint(config):
                raise RuntimeError('Checkpoint dở dang khác cấu hình. Dùng OUTPUT_ROOT mới.')
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(config['seed'])
    rows = select_training_entries(train_samples, config['train_fraction'], config['seed'])
    (directory / 'train_image_ids.json').write_text(json.dumps([e['image_id'] for e in rows]))
    (directory / 'loss.jsonl').write_text('')
    model = optimizer = scheduler = scaler = params = iterator = None
    history = []
    step = skipped = consecutive_skips = 0
    run_id = uuid.uuid4().hex
    try:
        model = load_base_model()
        model = apply_setup(model, config['setup'], config)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, int(config['max_steps'] * 0.05)),
            num_training_steps=config['max_steps'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        loader = DataLoader(RoadDataset(rows), batch_size=config['batch_size'], shuffle=True,
                            collate_fn=collate_fn, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed']))
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def record(complete):
            torch.cuda.synchronize()
            return {'config': config, 'fingerprint': config_fingerprint(config),
                    'run_id': run_id, 'complete': complete, 'completed_steps': step,
                    'skipped_steps': skipped, 'train_images': len(rows),
                    'trainable_params': sum(p.numel() for p in params),
                    'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
                    'train_minutes': (time.perf_counter() - started) / 60,
                    'dataset_version': DATASET_VERSION, 'model_id': MODEL_ID,
                    'model_revision': MODEL_COMMIT, 'prompt': PROMPT}

        while step < config['max_steps']:
            loss_sum = 0.0
            for _ in range(config['grad_accum']):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = to_device(batch)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(**batch)
                    loss = output.loss / config['grad_accum']
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Loss nan/inf: kiểm tra mask/dtype và giảm lr.')
                loss_sum += loss.detach().item()
                scaler.scale(loss).backward()
                del batch, output, loss
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            previous_scale = scaler.get_scale()
            used_lr = optimizer.param_groups[0]['lr']
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < previous_scale:
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips >= 8:
                    raise FloatingPointError('8 bước gradient overflow liên tiếp.')
                continue
            consecutive_skips = 0
            scheduler.step()
            step += 1
            entry = {'step': step, 'loss': loss_sum, 'lr': used_lr,
                     'grad_norm': float(norm), 'minutes': (time.perf_counter() - started) / 60}
            history.append(entry)
            with (directory / 'loss.jsonl').open('a') as stream:
                stream.write(json.dumps(entry) + '\n')
            if step == 1 or step % 10 == 0:
                print(f'{config["name"]}: {step}/{config["max_steps"]}, loss={loss_sum:.4f}')
            if step % 100 == 0 and step < config['max_steps']:
                save_checkpoint(model, directory / f'step-{step:04d}', record(False), history)
        save_checkpoint(model, final, record(True), history)
        return final
    finally:
        del model, optimizer, scheduler, scaler, params, iterator
        gc.collect()
        torch.cuda.empty_cache()


def restore_checkpoint(directory, config):
    directory = Path(directory)
    run = load_completed_run(directory, config)
    assert run is not None, f'Checkpoint chưa hoàn tất: {directory}'
    saved_processor = AutoProcessor.from_pretrained(directory / 'processor')
    assert saved_processor.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
    assert saved_processor.chat_template == processor.chat_template
    model = load_base_model()
    try:
        model = PeftModel.from_pretrained(model, directory / 'adapter', is_trainable=False)
        if config['train_projector']:
            state = torch.load(directory / 'extra_trainable.pt', map_location='cpu', weights_only=True)
            parameters = dict(model.get_base_model().named_parameters())
            assert set(state) == {n for n in parameters if is_projector(n)}
            with torch.no_grad():
                for name, value in state.items():
                    param = parameters[name]
                    assert value.shape == param.shape
                    param.data = param.data.float()
                    param.copy_(value.to(param.device))
                    assert torch.equal(param.detach().cpu(), value), name
            del parameters, state
        return model, run
    except Exception:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        raise


def infer_checkpoint(directory, config, split):
    run = load_completed_run(directory, config)
    assert run is not None
    if split == 'test':
        assert RUN_MODE == 'full' and run['completed_steps'] == 300
        rows = test_samples
    else:
        rows = val_samples[:VAL_LIMIT] if RUN_MODE == 'full' else val_samples[:3]
    path = prediction_path(config['name'], split)
    manifest = prediction_manifest(run, rows, split)
    if valid_prediction_file(path, rows, manifest):
        print('Dùng lại predictions:', path)
    else:
        model, run = restore_checkpoint(directory, config)
        try:
            predict_jsonl(model, rows, path, manifest)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if split == 'test':
        return score_predictions(path, config['name'], run)
    predictions = {r['image_id']: r['raw'] for r in read_jsonl(path)}
    result = evaluate(rows, predictions, n_boot=200)
    record = dict(result, fingerprint=run['fingerprint'], run_id=run['run_id'],
                  dataset_version=DATASET_VERSION, split='val', mode=RUN_MODE)
    path.with_suffix('.metrics.json').write_text(json.dumps(record, indent=2))
    backup_path(path.with_suffix('.metrics.json'))
    print(config['name'], 'val ảnh:', len(rows), 'F1:', result['micro']['f1'],
          'JSON valid:', result['parse_valid_rate'])
    for row in read_jsonl(path)[:3]:
        print(row['image_id'], row['raw'])
    return record


def execute_setup(name, config=None):
    config = config or load_config(name)
    checkpoint = train_experiment(config)
    validation = infer_checkpoint(checkpoint, config, 'val')
    test = infer_checkpoint(checkpoint, config, 'test') if RUN_MODE == 'full' else None
    return {'checkpoint': str(checkpoint), 'val': validation, 'test': test}


RESULTS = {}


def _bind_saved_artifacts():
    global FINGERPRINT_SCHEMA
    assert MODEL_COMMIT == MODEL_REVISION, 'Model commit tải về khác checkpoint cũ.'
    assert DATASET_VERSION == BASELINE_RUNS['E1']['dataset_version'], 'Dataset/split khác artifact cũ.'
    matches = []
    for schema in ('legacy', 'target_policy_v1'):
        FINGERPRINT_SCHEMA = schema
        if all(config_fingerprint(run['config']) == run['fingerprint']
               for run in BASELINE_RUNS.values()):
            matches.append(schema)
    if len(matches) != 1:
        FINGERPRINT_SCHEMA = None
        raise RuntimeError('Fingerprint không khớp hai phiên bản notebook đã biết. '
                           'Dừng; không bỏ qua hoặc sửa fingerprint của artifact.')
    FINGERPRINT_SCHEMA = matches[0]
    print('Fingerprint đã xác minh:', FINGERPRINT_SCHEMA)
    for name in BASELINE_RUNS:
        config = load_config(name)
        run = load_completed_run(run_directory(name) / 'final', config)
        assert run is not None
        for split in ('val', 'test'):
            rows = val_samples[:VAL_LIMIT] if split == 'val' else test_samples
            path = prediction_path(name, split)
            expected = prediction_manifest(run, rows, split)
            if not valid_prediction_file(path, rows, expected):
                raise RuntimeError(f'{name}/{split}: predictions hoặc manifest không khớp run cũ.')
        metrics = json.loads(prediction_path(name, 'val').with_suffix('.metrics.json').read_text())
        if (metrics.get('run_id') != run['run_id'] or metrics.get('fingerprint') != run['fingerprint']
                or metrics.get('n_images') != min(VAL_LIMIT, len(val_samples))):
            raise RuntimeError(f'{name}: val metrics không khớp run cũ.')
        evidence = json.loads((EVIDENCE_ROOT / f'{name.lower()}.json').read_text())
        if (evidence.get('dataset_version') != DATASET_VERSION or evidence.get('split') != 'test'
                or evidence.get('n_images') != len(test_samples)
                or evidence.get('missing_predictions') != 0
                or evidence.get('meta', {}).get('run_id') != run['run_id']
                or evidence.get('meta', {}).get('fingerprint') != run['fingerprint']):
            raise RuntimeError(f'{name}: evidence không khớp predictions/checkpoint.')
        print(name, ': checkpoint, predictions, metrics và evidence hợp lệ; không train/infer lại.')

_bind_saved_artifacts()
# Kiểm tra toàn bộ target trước khi nạp trọng số GPU cho E4.
for _entry in train_samples:
    assert parse_output(objects_to_answer(_entry['objects'])) is not None


def choose_e4_base():
    if E4_BASE_SETUP != 'auto':
        return E4_BASE_SETUP
    candidates = []
    for name in ('E1', 'E2', 'E3'):
        config = load_config(name)
        directory = run_directory(name) / 'final'
        run = load_completed_run(directory, config)
        metrics_path = prediction_path(name, 'val').with_suffix('.metrics.json')
        if run and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            expected_n = min(len(val_samples), VAL_LIMIT if RUN_MODE == 'full' else 3)
            expected_rows = val_samples[:expected_n]
            expected_manifest = prediction_manifest(run, expected_rows, 'val')
            if (metrics.get('fingerprint') == run['fingerprint']
                    and metrics.get('run_id') == run['run_id'] and metrics['n_images'] == expected_n
                    and valid_prediction_file(prediction_path(name, 'val'), expected_rows, expected_manifest)):
                candidates.append((metrics['micro']['f1'], -run['peak_vram_gb'], name))
    assert candidates, 'Cần chạy val E1/E2/E3 hoặc khôi phục run cũ từ Drive trước E4.'
    print('E4 candidates (val F1, -VRAM, setup):', candidates)
    return max(candidates)[2]


if 'E4' in RUNS:
    base_name = choose_e4_base()
    base_config = load_config(base_name)
    base_directory = run_directory(base_name) / 'final'
    base_run = load_completed_run(base_directory, base_config)
    assert base_run is not None
    if RUN_MODE == 'full':
        baseline_predictions = prediction_path(base_name, 'test')
        assert valid_prediction_file(baseline_predictions, test_samples,
                                     prediction_manifest(base_run, test_samples, 'test')), (
            'Cần hoàn tất test của baseline với cấu hình inference hiện tại trước E4.')
    small_config = copy.deepcopy(base_config)
    small_config.update(name='E4_data_size_0.1', train_fraction=0.1)
    RESULTS['E4_data_size_0.1'] = execute_setup('E4_data_size_0.1', small_config)
    if RUN_MODE == 'full':
        baseline_predictions = prediction_path(base_name, 'test')
        assert valid_prediction_file(baseline_predictions, test_samples,
                                     prediction_manifest(base_run, test_samples, 'test'))
        RESULTS['E4_data_size_1.0'] = score_predictions(
            baseline_predictions, 'E4_data_size_1.0', base_run)
    (OUTPUT_ROOT / f'e4_selection_{RUN_MODE}.json').write_text(
        json.dumps({'base_setup': base_name, 'variable': 'data_size', 'values': [0.1, 1.0],
                    'val_limit': VAL_LIMIT, 'base_fingerprint': base_run['fingerprint']}, indent=2))

if 'E4' in RUNS:
    backup_path(OUTPUT_ROOT / f'e4_selection_{RUN_MODE}.json')


print('E4 đã hoàn tất và backup. Chạy các cell bảng kết quả/ảnh/xuất báo cáo bên dưới.')
