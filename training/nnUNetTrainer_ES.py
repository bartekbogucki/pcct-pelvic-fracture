import torch
import numpy as np
from os.path import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_ES(nnUNetTrainer):
    """
    nnU-Net trainer with:
      * Early stopping on validation loss (patience = 100 epochs).
      * Saves THREE extra 'best' checkpoints:
          - checkpoint_best_valloss.pth   (lowest val_loss)   <- the one we report
          - checkpoint_best_emadice.pth   (highest EMA pseudo Dice)
          - checkpoint_best_rawdice.pth   (highest raw mean fg Dice)
      * nnU-Net's own checkpoint_best.pth (EMA-Dice) is still saved by the parent.
      * Auto-validation after training is DISABLED (perform_actual_validation no-op),
        because we evaluate separately with the val_loss checkpoint using our own
        eval script (Dice / surface Dice 1mm+2mm / recall / HD95). This avoids the
        slow ~2.5h/fold auto-validation that uses the EMA checkpoint we don't report.

    Use:  -tr nnUNetTrainer_ES
    """
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.es_patience    = 100
        self._best_valloss  = np.inf
        self._best_es_ema   = -np.inf
        self._best_rawdice  = -np.inf
        self._epochs_since_improve = 0
        self._stop_early    = False

    def on_epoch_end(self):
        # Read metrics BEFORE super() (parent increments current_epoch at its end).
        log = self.logger.my_fantastic_logging
        val_loss = float(log['val_losses'][-1])
        ema      = float(log['ema_fg_dice'][-1])
        rawdice  = float(log['mean_fg_dice'][-1])
        epoch    = self.current_epoch

        super().on_epoch_end()   # parent: logs, plots, saves checkpoint_best.pth (EMA), epoch++

        # ---- save best-of-three ----
        improved_valloss = val_loss < self._best_valloss - 1e-6
        if improved_valloss:
            self._best_valloss = val_loss
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_valloss.pth'))
            self.print_to_log_file(f"[ES] new best val_loss {val_loss:.4f} (epoch {epoch})")

        if ema > self._best_es_ema + 1e-6:
            self._best_es_ema = ema
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_emadice.pth'))

        if rawdice > self._best_rawdice + 1e-6:
            self._best_rawdice = rawdice
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_rawdice.pth'))
            self.print_to_log_file(f"[ES] new best raw Dice {rawdice:.4f} (epoch {epoch})")

        # ---- early stopping on val_loss ----
        if improved_valloss:
            self._epochs_since_improve = 0
        else:
            self._epochs_since_improve += 1

        if self._epochs_since_improve >= self.es_patience:
            self.print_to_log_file(
                f"[ES] val_loss has not improved for {self.es_patience} epochs "
                f"(best {self._best_valloss:.4f}). Stopping after epoch {epoch}.")
            self._stop_early = True

    def run_training(self):
        # while-loop version that honors _stop_early (parent's for-range can't be interrupted)
        self.on_train_start()

        while self.current_epoch < self.num_epochs and not self._stop_early:
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()   # may set self._stop_early

        self.on_train_end()

    def perform_actual_validation(self, save_probabilities: bool = False):
        # Skip nnU-Net's built-in post-training validation/prediction.
        # We evaluate separately with checkpoint_best_valloss.pth (our reported
        # criterion) using our own eval script, computing Dice / surface Dice
        # (1mm & 2mm) / recall / HD95. The default auto-validation predicts with
        # checkpoint_best.pth (EMA), which we are not reporting, and takes ~2.5h/fold.
        self.print_to_log_file(
            "[ES] Skipping built-in auto-validation. Evaluate separately with "
            "checkpoint_best_valloss.pth via the standalone eval script.")
        return
