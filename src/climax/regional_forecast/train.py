# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import pandas as pd

from climax.regional_forecast.datamodule import RegionalForecastDataModule
from climax.regional_forecast.module import RegionalForecastModule
from pytorch_lightning.cli import LightningCLI


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=RegionalForecastModule,
        datamodule_class=RegionalForecastDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # Force root_device to GPU to avoid the error
    if torch.cuda.is_available():
        cli.trainer = cli.trainer.__class__(
        accelerator="gpu",
        devices=1,
        precision=16,
        default_root_dir=cli.trainer.default_root_dir,
        callbacks=cli.trainer.callbacks,
        logger=cli.trainer.logger,
        max_epochs=cli.trainer.max_epochs,
    )

    cli.datamodule.set_patch_size(cli.model.get_patch_size())

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    torch.cuda.empty_cache()

    # fit() runs the training
    checkpoint_path = "/workspace/climax_logs/checkpoints/last.ckpt"

    if os.path.exists(checkpoint_path):
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=checkpoint_path)
        print(f"✅ Resumed training from checkpoint: {checkpoint_path}")
    else:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        print("🚀 Starting training from scratch (no checkpoint found).")

    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")

    # # Run test and save results
    # test_results = cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")
    
    # os.makedirs(f"{cli.trainer.default_root_dir}/metrics", exist_ok=True)
    # results_path = os.path.join(cli.trainer.default_root_dir, "metrics", "test_metrics_3days_10epochs.csv")
    # pd.DataFrame(test_results).to_csv(results_path, index=False)
    # print(f"\n✅ Test metrics saved to: {results_path}")


if __name__ == "__main__":
    main()
