"""SFT-only entry point for the retained LLaMA-Factory runtime."""
from ..hparams import get_train_args, read_args
from .callbacks import LogCallback, ReporterCallback
from .sft import run_sft

def main():
    args = get_train_args(read_args())
    model, data, training, finetuning, generating = args
    if finetuning.stage != "sft" or finetuning.finetuning_type != "full":
        raise ValueError("The release supports full SFT only")
    callbacks = [LogCallback(), ReporterCallback(model, data, finetuning, generating)]
    run_sft(*args, callbacks=callbacks)

if __name__ == "__main__":
    main()
