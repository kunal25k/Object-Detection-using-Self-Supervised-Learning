Please run sbatch demo.slurm - to run this.
This runs "threestage.py" which generates the cropped images into /generated/images/ and labels into /generated/labels ( in the same semsl2 folder but we removed this since data was about 40 gb - 660k images )
We took these images and then geenrated unlabeledcrop which is fed into barlow_cropdata_training