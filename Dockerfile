# Use a base image with the desired operating system and dependencies
FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
COPY requirements.txt .

RUN conda create -n semantic_mapping -f environment.yml && \
    conda clean -a -y && \
    echo "conda activate semantic_mapping" >> ~/.bashrc
RUN echo "source activate semantic_mapping" > ~/.bashrc
ENV PATH /opt/conda/envs/semantic_mapping/bin:$PATH
RUN pip install -r requirements.txt

COPY . .

CMD ["conda", "activate", "semantic_mapping"]

