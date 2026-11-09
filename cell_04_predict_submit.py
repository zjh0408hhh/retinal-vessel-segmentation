# 第4部分：测试集预测和提交文件生成
# 已有权重和配置文件时，可按第1、2、4部分的顺序重新预测。
config = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
assert config['base'] == BASE, 'BASE 必须与训练时一致'
all_checkpoints = config['checkpoints']
assert len(all_checkpoints) == N_FOLDS * len(PLANS) == 3, '本版本只使用三个模型'
THRESHOLD, MIN_COMPONENT = config['threshold'], config['min_component']
test_features, test_fovs = [], []
for sid in test_ids:
    x, fov = make_features(read_rgb(test_files[sid]))
    test_features.append(x)
    test_fovs.append(fov)
test_prob = [np.zeros(x.shape[:2], np.float32) for x in test_features]
print('开始集成预测。', flush=True)
for entry in all_checkpoints:
    model = VesselNet().to(DEVICE)
    model.load_state_dict(torch.load(entry['path'], map_location=DEVICE, weights_only=True))
    for i, features in enumerate(test_features):
        test_prob[i] += predict_native(model, features, entry['patch'], tta=True) / len(all_checkpoints)
    del model
    if AMP:
        torch.cuda.empty_cache()

mask_dir = RUN / 'image'
encoder_root = RUN / 'encoding'
encoder_dir = encoder_root / 'image'
mask_dir.mkdir(parents=True, exist_ok=True)
encoder_dir.mkdir(parents=True, exist_ok=True)
expected = {f'{sid}.png' for sid in test_ids}
extra = {p.name for p in encoder_dir.iterdir() if p.is_file()} - expected
assert not extra, f'编码目录有额外文件：{extra}'
vessels = {}
for i, sid in enumerate(test_ids):
    p = test_prob[i] * test_fovs[i]
    vessel = output_mask(p, THRESHOLD, MIN_COMPONENT)
    vessels[sid] = vessel
    # 保存黑色血管图，另存一份白色血管图供官方脚本编码。
    Image.fromarray(np.where(vessel, 0, 255).astype(np.uint8)).save(mask_dir / f'{sid}.png')
    Image.fromarray(vessel.astype(np.uint8) * 255).save(encoder_dir / f'{sid}.png')
np.savez_compressed(RUN / 'test_native_probability.npz',
                    **{f'prob_{sid}': test_prob[i] * test_fovs[i] for i, sid in enumerate(test_ids)})
subprocess.run([sys.executable, str(ROOT / 'segmentation_to_csv.py')],
               cwd=str(encoder_root), check=True)
submission = pd.read_csv(encoder_root / 'submission.csv',
                         dtype={'Id': str}, keep_default_na=False)
assert list(submission.columns) == ['Id', 'Predicted']
assert len(submission) == 20 and submission['Id'].is_unique
assert set(submission['Id']) == set(test_ids)
submission = submission.set_index('Id').loc[test_ids].reset_index()

def decode_and_check(text):
    parts = str(text).split()
    assert len(parts) % 2 == 0
    flat = np.zeros(512 * 512, np.uint8)
    previous_end = 0
    for a, b in zip(parts[::2], parts[1::2]):
        start, length = int(a), int(b)
        end = start + length - 1
        assert start > previous_end and start > 0 and length > 0
        assert end <= flat.size
        flat[start - 1:end] = 1
        previous_end = end
    return flat.reshape(512, 512)

# 解码检查CSV中的血管像素是否与预测一致。
for row in submission.itertuples(index=False):
    restored = decode_and_check(row.Predicted)
    assert np.array_equal(restored, vessels[row.Id]), f'前景或编码错误：{row.Id}'
ratio = float(np.mean([v.mean() for v in vessels.values()]))
assert 0 < ratio < 0.5, f'血管占比异常：{ratio:.4f}'
submission.to_csv(RUN / 'submission.csv', index=False)
submission.to_csv(OUT / 'submission.csv', index=False)
print(f'检查通过：20 行、顺序正确、正数、无重叠、回解为血管；血管占比={ratio:.4f}')
print('提交文件：/kaggle/working/submission.csv')
print('黑色血管预测图：', mask_dir)
display(submission.head())
