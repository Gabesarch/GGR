"""Linear LR decay with the minimum LR applied on the final training update."""

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """Step once after each training batch; scheduler state resumes via LambdaLR."""
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if not 0 <= num_warmup_steps < num_training_steps - 1:
        raise ValueError("Linear decay requires at least two post-warmup training updates")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        # LambdaLR initializes at 0 and advances after each training update.
        progress = min(1.0, (current_step - num_warmup_steps) / (num_training_steps - 1 - num_warmup_steps))
        return 1.0 - (1.0 - min_lr_ratio) * progress

    return LambdaLR(optimizer, lr_lambda)
