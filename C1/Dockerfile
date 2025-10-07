FROM quay.io/jupyter/scipy-notebook:python-3.12

USER root

RUN apt-get update

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt && rm requirements.txt 

# COPY jupyter_notebook_config.py /etc/jupyter/
COPY .jupyter/ /home/$NB_USER/.jupyter/
RUN fix-permissions /home/$NB_USER/.jupyter
COPY overrides.json ${CONDA_DIR}/share/jupyter/lab/settings/overrides.json
ENV PYTHONWARNINGS ignore
ENV IN_COURSERA_ENVIRON true

# Set entry point
EXPOSE 8888
WORKDIR /home/$NB_USER/work
ENTRYPOINT ["tini", "--"]
CMD ["start-notebook.sh"]

USER $NB_USER
