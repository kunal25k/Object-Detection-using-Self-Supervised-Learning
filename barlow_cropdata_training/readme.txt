This training is done on the unlabeledext set which is basically the cropped regions form the original images where we expect the original image to be there
Please run sbatch demodist.slurm - to train this
The model output weights ( our ssl backbone ) are stored in checkpoint folder as resnet50_{epochnum}.pth ( for every 10 epochs )
The output is printed to stats.txt in the checkpoint folder