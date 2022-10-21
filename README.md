# Plankton Domain Adaptation





## Usage

## Notes

On a machine with a Nvidia GTX3080, 32 is the maximum batch size (OOM with batch size of 64)

You may come across the following error with batch sizes that != 0 %32 (this isn't entirely accuracte: find underlying condition)

RuntimeError: Unable to find a valid cuDNN algorithm to run convolution

