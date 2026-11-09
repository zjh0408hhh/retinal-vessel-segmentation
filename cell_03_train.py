# 第3部分：三折训练和阈值选择
all_checkpoints, history, summaries = [], [], []
oof_by_plan = [[None for _ in train_ids] for _ in PLANS]
for plan_id, plan in enumerate(PLANS):
    patch = plan['patch']
    for fold, val_idx in enumerate(folds):
        seed = plan['seed'] + fold
        seed_all(seed)
        train_idx = np.setdiff1d(np.arange(len(train_ids)), val_idx)
        assert not set(train_idx) & set(val_idx)
        loader = DataLoader(VesselPatches(train_idx, patch), batch_size=BATCH,
                            shuffle=True, num_workers=WORKERS, pin_memory=AMP,
                            persistent_workers=(WORKERS > 0), worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(seed))
        model = VesselNet().to(DEVICE)
        ema = EMA(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=AMP)
        path = RUN / f'ema_patch{patch}_fold{fold}.pt'
        best, stale, best_epoch = -1.0, 0, 0
        best_train_loss = best_val_loss = float('inf')
        print(f'Model {plan_id + 1}/{len(PLANS)} | patch={patch} | Fold {fold + 1}', flush=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_sum = 0.0
            for x, y in loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                train_sum += float(loss.detach())
            scheduler.step()
            train_loss = train_sum / len(loader)
            score, val_loss = validate(ema.model, val_idx, patch)
            # 保存验证Dice最高的EMA权重。
            if score > best:
                best, stale, best_epoch = score, 0, epoch
                best_train_loss, best_val_loss = train_loss, val_loss
                torch.save(ema.model.state_dict(), path)
            else:
                stale += 1
            if epoch % LOG_EVERY == 0 or epoch == EPOCHS:
                print(f'Fold {fold + 1} | epoch {epoch:03d} | '
                      f'血管 Dice={score:.4f} | best={best:.4f}', flush=True)
            history.append({'plan': plan_id + 1, 'patch': patch, 'fold': fold + 1,
                            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                            'vessel_dice_official_resize': score, 'best': best})
            if stale >= PATIENCE and epoch >= MIN_EPOCHS:
                break
        # 记录最佳轮次对应的训练损失和验证损失。
        print(f'Fold {fold + 1} | best={best:.4f} | loss={best_val_loss:.4f} | '
              f'best_epoch={best_epoch:03d} | train_loss={best_train_loss:.4f}', flush=True)
        summaries.append({'plan': plan_id + 1, 'patch': patch, 'fold': fold + 1,
                           'best': best, 'best_epoch': best_epoch,
                           'val_bce_dice_at_best': best_val_loss,
                           'train_loss_at_best': best_train_loss})
        pd.DataFrame(history).to_csv(RUN / 'training_history.csv', index=False)
        pd.DataFrame(summaries).to_csv(RUN / 'fold_summary.csv', index=False)
        ema.model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        for i in val_idx:
            oof_by_plan[plan_id][i] = predict_native(ema.model, images[i], patch, tta=True) * fovs[i]
        all_checkpoints.append({'path': str(path), 'patch': patch, 'fold': fold,
                                 'plan': plan_id})
        del loader, model, ema, optimizer, scheduler, scaler
        gc.collect()
        if AMP:
            torch.cuda.empty_cache()

# 用各折的验证预测选择提交阈值。
oof = [np.mean([oof_by_plan[p][i] for p in range(len(PLANS))], axis=0)
       for i in range(len(train_ids))]
indices = np.arange(len(train_ids))
calibration = []
for min_size in [0, 8]:
    for threshold in np.arange(0.25, 0.751, 0.025):
        score = evaluate_probabilities(oof, indices, float(threshold), min_size)
        calibration.append({'threshold': float(threshold), 'min_size': min_size, 'dice': score})
best_setting = max(calibration, key=lambda row: (row['dice'], -abs(row['threshold'] - 0.5)))
without_filter = max((r for r in calibration if r['min_size'] == 0), key=lambda r: r['dice'])
if best_setting['dice'] < without_filter['dice'] + 0.001:
    best_setting = without_filter  # Dice提升不到0.001时保留小连通域。
THRESHOLD, MIN_COMPONENT = best_setting['threshold'], best_setting['min_size']
native_dice = float(np.mean([dice_one(remove_small(p > THRESHOLD, MIN_COMPONENT), masks[i])
                            for i, p in enumerate(oof)]))
print(f"折外验证：转换口径 Dice={best_setting['dice']:.4f} | "
      f'原分辨率 Dice={native_dice:.4f} | threshold={THRESHOLD:.3f} | min_size={MIN_COMPONENT}')
print('转换口径依据附件脚本；与旧版最近邻标签的 Dice 不可直接比较，也不是平台测试分数。')
pd.DataFrame(calibration).to_csv(RUN / 'oof_calibration.csv', index=False)
np.savez_compressed(RUN / 'oof_native.npz',
                    **{f'prob_{sid}': oof[i].astype(np.float32) for i, sid in enumerate(train_ids)})
config = {'seed': SEED, 'plans': PLANS, 'folds': N_FOLDS, 'epochs_max': EPOCHS,
           'samples_per_epoch': SAMPLES_PER_EPOCH, 'base': BASE, 'batch': BATCH,
           'threshold': THRESHOLD, 'min_component': MIN_COMPONENT,
           'oof_dice_official_resize': best_setting['dice'], 'oof_dice_native': native_dice,
           'mask_resize': 'Pillow BILINEAR then >0, matching provided converter',
           'checkpoints': all_checkpoints, 'external_data': False,
           'pretrained_weights': False, 'rle_target': 'vessel_foreground'}
(RUN / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


