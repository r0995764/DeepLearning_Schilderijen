#!/bin/bash -l
#SBATCH --cluster=genius                        # We gebruiken de wICE cluster (Slide 18)
#SBATCH --partition=gpu_p100                    # We willen een GPU (Slide 83)
#SBATCH --gpus-per-node=1                       # 1 GPU kaart
#SBATCH --nodes=1                               # 1 Computer
#SBATCH --time=00:30:00                         # Max 30 minuten (pas aan indien nodig)
#SBATCH --account=lp_edu_vives_ti_deep_learning
#SBATCH --ntasks=9

# 1. Modules laden (Slide 64)
# We laden TensorFlow en Python.
module load TensorFlow/2.13.0-foss-2023a
module load matplotlib/3.7.2-gfbf-2023a

# 2. Naar de juiste map gaan
# $VSC_DATA is waar je bestanden staan (Slide 27)
WORKDIR=$VSC_SCRATCH/Schilderijen_Project
mkdir -p $WORKDIR
cd $WORKDIR
cp $VSC_DATA/Schilderijen_Project/dataset_processed_vsc.zip .

# 3. Zip uitpakken (als dat nog niet gebeurd is)
echo "Uitpakken dataset..."
unzip -q -o dataset_processed_vsc.zip

# 4. Je Python script starten
echo "Start training..."
python $VSC_DATA/Schilderijen_Project/train_vsc.py
python $VSC_DATA/Schilderijen_Project/train_augmentation.py

cp baseline_model.keras baseline_result.png $VSC_DATA/Schilderijen_Project/
cp augmented_model.keras augmented_result.png $VSC_DATA/Schilderijen_Project/



