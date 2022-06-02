
Our final submitted model involved four stages of training including fine-tuning:

First stage: ( using demo files )
Input: labeled data
Method: Running basic fastrcnn method on labeled data
output: fastrcnn.pth
How to run: demo-fastrcnn.slurm

Second Stage: ( semsl2 folder )
Input: fastrcnn.pth ( while training was named as gcnfastrcnn_13.pth ) - still included in the folder
Method: Run the model on unlabeled data, generate object detection regions, apply custom algo to get the best crops and store the crops
output: cropped unlabeled data stored in generated/images folder and and labels into /generated/labels in the same semsl2 folder ( but we removed this since data was about 40 gb ~ 660k images )
How to run: "sbatch demo.slurm" in the semsl2 folder

Processing:
I took these images and squashed them into unlabeledcrop.sqf and gave that path in the third stage which was /scratch/sca321/dlproject/data/unlabeledcrop.sqf

Third stage: ( barlow_cropdata_training folder ) 
Input: Cropped unlabeled datavfrom the above stage
Method: Run Barlow Twins on this cropped data
Output: The model output weights ( our ssl backbone ) are stored in checkpoint folder as resnet50_{epochnum}.pth ( for every 10 epochs ) - we removed them since they were heavy and uploaded one model - barlowcrop50.pth( trained for 50 epochs ). The output stats are printed to stats.txt in the checkpoint folder
How to run: "sbatch demodist.slurm" in the barlow_cropdata_training folder which runs demodist.py


Fourth stage: ( finalfinetune folder )
Input: barlowcrop50.pth - still included in the folder
Method: fine-tune the fasterrcnn model with the backbone replaced by the above input model
Output: weights.pth ( final submitted weights ) and demo_{jobbed}.out files have the statistics.
How to run: "sbatch demo.slurm" in finalfinetune folder


Alternate Models:
BYOL:
Input: Cropped unlabeled datavfrom the above stage
Method: Run BYOL on this Unlabeled data
Output: The model output weights ( our ssl backbone ) are stored in checkpoint folder as SelfSupervised.h5 ( for every 20 epochs )
How to run: "sbatch demo_train.slurm"
