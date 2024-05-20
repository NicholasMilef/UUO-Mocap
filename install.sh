conda create -n video_mocap python=3.10
conda activate video_mocap

# install PyTorch and PyTorch3D
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d

# install pip and conda packages
pip install joblib
pip install opencv-python
pip install trimesh[easy]
pip install pyrender
pip install roma
pip install pybullet
python -m pip install libigl
pip install mergedeep
pip install git+https://github.com/mattloper/chumpy.git
conda install -y -c conda-forge ezc3d
conda install -y matplotlib
conda install -y -c conda-forge scikit-learn
conda install -y seaborn

# install MoSh++, SMPL-X, and Human Body Prior
cd third_party/moshpp/
pip install -e .
cd ..

cd third_party/human_body_prior/
pip install -e .
cd ..

cd third_party/smplx/
pip install -e .
cd ..

# install this repository as a package
pip install -e .