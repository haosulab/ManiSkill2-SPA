FROM haosulab/mani-skill2:latest

# Install additional python packages you need

# Copy your codes (including user_solution.py) and model weights
COPY ./ /root/
ENV PYTHONPATH ./:$PYTHONPATH
RUN conda init bash
RUN conda create --name ms2tpn python=3.8
SHELL ["conda", "run", "-n", "ms2tpn", "/bin/bash", "-c"]
RUN cd /root/ravens && pip install -r requirements.txt && python setup.py install --user

RUN conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0 && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ && \
    pip install --upgrade pip && \
    pip install tensorflow==2.11.* && \
    pip install tensorflow-addons==0.19.0 && \
    conda install pinocchio -c defaults -c conda-forge && \
    pip install --upgrade git+https://github.com/Jiayuan-Gu/pymp.git

RUN echo "source activate ms2tpn" > ~/.bashrc