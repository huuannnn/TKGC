pip install -r requirements.txt && \
pip install torch_geometric && \
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.11.0+cu126.html && \
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html && \
pip uninstall -y torch torchvision torchaudio && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121