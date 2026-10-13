"""CPU checks for the authored notebook. Does not modify instructor scripts/tests.

Run: python3 notebook_checks.py
CUDA/model execution still needs a Kaggle smoke run.
"""
import ast
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import evaluate, iou_xyxy, parse_output


def main():
    notebook = json.loads(Path('lab_day10.ipynb').read_text())
    definitions = {}
    calls = []
    constants = {}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        assert cell['outputs'] == [] and cell['execution_count'] is None
        source = ''.join(cell['source'])
        assert 'google.colab' not in source and '/content/' not in source and 'DRIVE_ROOT' not in source
        tree = ast.parse(source, filename=f'cell_{index}')
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = node
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node)
    assert len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
    assert constants['RUN_MODE'] == 'full'
    assert constants['RUNS'] == ('E0', 'E1', 'E2', 'E3', 'E4')
    assert constants['TRAIN_BATCH_SIZE'] * constants['GRAD_ACCUM'] == 8
    assert constants['E4_BASE_SETUP'] == 'auto'
    assert any(n.func.id == 'execute_setup' and isinstance(n.args[0], ast.Constant)
               and n.args[0].value == 'E2' for n in calls)
    assert any(n.func.id == 'execute_setup' and isinstance(n.args[0], ast.Constant)
               and n.args[0].value == 'E3' for n in calls)
    assert any(n.func.id == 'BitsAndBytesConfig' and
               any(k.arg == 'llm_int8_skip_modules' and
                   ast.literal_eval(k.value) == ['model.visual', 'lm_head'] for k in n.keywords)
               for n in calls)
    packages = [line for line in Path('requirements-kaggle.txt').read_text().splitlines()
                if line and not line.startswith('#')]
    assert packages == constants['PINNED'] + constants['EXTRAS']

    ns = dict(json=json, hashlib=hashlib, random=random, Path=Path, re=re,
              CLASSES=CLASSES, read_jsonl=read_jsonl, parse_output=parse_output, iou_xyxy=iou_xyxy)
    selected = ['yolo_to_box', 'objects_to_answer', 'lora_pattern', 'is_projector',
                'fingerprint', 'prediction_manifest', 'valid_prediction_file',
                'select_training_entries', 'compute_reward', 'discover_dataset',
                'config_fingerprint', 'load_completed_run']
    for name in selected:
        exec(compile(ast.Module(body=[definitions[name]], type_ignores=[]), 'notebook', 'exec'), ns)
    convert, answer = ns['yolo_to_box'], ns['objects_to_answer']
    assert answer([]) == '[]'
    assert convert('pothole', [.5, .5, .2, .1])['bbox_2d'] == [400, 450, 600, 550]
    assert convert('crack', [.02, .02, .1, .1])['bbox_2d'] == [0, 0, 70, 70]
    objects = [{'label': 'crack', 'yolo': [.8, .8, .1, .1]},
               {'label': 'pothole', 'yolo': [.4, .2, .1, .1]},
               {'label': 'manhole', 'yolo': [.2, .2, .1, .1]}]
    assert [o['label'] for o in json.loads(answer(objects))] == ['manhole', 'pothole', 'crack']
    entries = [{'image_id': str(i), 'width': 640, 'height': 360, 'objects': objects}
               for i in range(20)]
    metrics = evaluate(entries, {e['image_id']: answer(e['objects']) for e in entries}, n_boot=40)
    assert metrics['micro']['f1'] == metrics['parse_valid_rate'] == 1
    assert metrics['missing_predictions'] == 0

    names = []
    for i in range(36):
        names += [f'model.language_model.layers.{i}.self_attn.{p}_proj' for p in ('q', 'k', 'v', 'o')]
        names += [f'model.language_model.layers.{i}.mlp.{p}_proj' for p in ('gate', 'up', 'down')]
    for i in range(24):
        names += [f'model.visual.blocks.{i}.attn.{p}' for p in ('qkv', 'proj')]
        names += [f'model.visual.blocks.{i}.mlp.{p}' for p in ('linear_fc1', 'linear_fc2')]
    names += ['model.visual.merger.linear_fc1', 'model.visual.deepstack_merger_list.0.linear_fc2',
              'model.visual.patch_embed.proj', 'lm_head']
    for setup, expected in [('E1', 252), ('E2', 252), ('E3', 348)]:
        matches = [name for name in names if re.fullmatch(ns['lora_pattern'](setup), name)]
        assert len(matches) == expected
        assert not any(ns['is_projector'](name) for name in matches)
    assert 'model.visual.blocks.0.attn.proj' in matches

    sample = ns['select_training_entries'](entries, .1, 42)
    assert len(sample) == 2
    assert sample == ns['select_training_entries'](entries, 1, 42)[:2]
    assert sample == ns['select_training_entries'](entries, .1, 42)
    try:
        ns['select_training_entries'](entries, 0, 42)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid fraction accepted')

    ns['MAX_NEW_TOKENS'] = 256
    run = {'run_id': 'r1', 'fingerprint': 'f1'}
    manifest = ns['prediction_manifest'](run, entries, 'test')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'pred.jsonl'
        metadata = path.with_suffix('.manifest.json')
        metadata.write_text(json.dumps(manifest))
        rows = [{'image_id': e['image_id'], 'raw': answer(e['objects'])} for e in entries]

        def write_rows(value):
            path.write_text(''.join(json.dumps(row) + '\n' for row in value))

        write_rows(rows)
        assert ns['valid_prediction_file'](path, entries, manifest)
        assert not ns['valid_prediction_file'](path, entries, dict(manifest, run_id='r2'))
        assert not ns['valid_prediction_file'](path, entries, dict(manifest, max_new_tokens=512))
        write_rows(rows[:-1])
        assert not ns['valid_prediction_file'](path, entries, manifest)
        write_rows(rows[:-1] + [rows[0]])
        assert not ns['valid_prediction_file'](path, entries, manifest)
        write_rows(rows[:-1] + [dict(rows[-1], raw=None)])
        assert not ns['valid_prediction_file'](path, entries, manifest)
        root = Path(tmp) / 'input'
        (root / 'nested/dataset/images').mkdir(parents=True)
        (root / 'nested/dataset/labels-YOLO').mkdir()
        assert ns['discover_dataset'](root) == root / 'nested/dataset'
        (root / 'other/images').mkdir(parents=True)
        (root / 'other/labels-YOLO').mkdir()
        try:
            ns['discover_dataset'](root)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Ambiguous dataset accepted')

        ns.update(MODEL_ID='test-model', MODEL_COMMIT='test-commit',
                  DATASET_VERSION='test-dataset', PROMPT='test-prompt', RUN_MODE='full',
                  PINNED=constants['PINNED'])
        cfg = {'name': 'E2', 'setup': 'E2', 'train_projector': True, 'max_steps': 300}
        checkpoint = Path(tmp) / 'checkpoint'
        (checkpoint / 'adapter').mkdir(parents=True)
        record_path = checkpoint / 'run.json'
        run = {'fingerprint': ns['config_fingerprint'](cfg), 'complete': True,
               'completed_steps': 300}
        record_path.write_text(json.dumps(run))
        (checkpoint / 'adapter/adapter_config.json').write_text('{}')
        (checkpoint / 'adapter/adapter_model.safetensors').write_bytes(b'fixture')
        try:
            ns['load_completed_run'](checkpoint, cfg)
        except RuntimeError as exc:
            assert 'thiếu projector' in str(exc)
        else:
            raise AssertionError('E2 checkpoint missing projector accepted')
        (checkpoint / 'extra_trainable.pt').write_bytes(b'fixture')
        assert ns['load_completed_run'](checkpoint, cfg) == run
        record_path.write_text(json.dumps(dict(run, complete=False, completed_steps=100)))
        assert ns['load_completed_run'](checkpoint, cfg) is None
        record_path.write_text(json.dumps(run))
        ns['RUN_MODE'] = 'smoke'
        try:
            ns['load_completed_run'](checkpoint, cfg)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Full/smoke checkpoint namespace collision')
        ns['RUN_MODE'] = 'full'
        changed = dict(cfg, max_steps=2)
        try:
            ns['load_completed_run'](checkpoint, changed)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Different training config accepted as same checkpoint')

    reward = ns['compute_reward']
    box = {'label': 'pothole', 'bbox': [100, 100, 200, 200]}
    perfect = json.dumps([{'label': 'pothole', 'bbox_2d': box['bbox']}])
    duplicate = json.dumps([{'label': 'pothole', 'bbox_2d': box['bbox']}] * 2)
    assert reward(perfect, [box]) == 1
    assert reward('broken', [box]) == 0
    assert reward('[]', []) == 1
    assert reward('[]', [box]) == .2
    assert abs(reward(duplicate, [box]) - .6) < 1e-9
    print('PASS: syntax, Kaggle setup, E2/E3 cells, labels, LoRA scope, sampling, checkpoint/prediction reuse and reward.')


if __name__ == '__main__':
    main()
