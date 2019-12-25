# Use python, version 3.6 as the base image.
# We won't install debian packages, so just use the slim variant.
FROM python:3.6-slim

# Install required python packages
# Note: This way of formating the instruction allows to easily
# add/remove/comment packages


#RUN pip install --trusted-host pypi.python.org -r	requirements.txt
RUN pip install --no-cache-dir \
        jupyter==1.0.0\
        tensorflow==1.14.0\
        Pillow==6.2.0\
        scikit-image==0.15.0\
        numpy==1.17.2\
        pyyaml==5.1.2\
        matplotlib==3.1.1\
	xlrd\
        ;

# Use /work as the working directory
RUN mkdir -p /work
WORKDIR /work
RUN mkdir -p /work/images
RUN mkdir -p /work/images/train
RUN mkdir -p /work/images/target
RUN mkdir -p /work/images/preprocessed




# Include the notebook
ADD config.yml  /work/config.yml
ADD model.py /work/model.py
ADD imageProcessor.py /work/imageProcessor.py
ADD train3.py /work/train3.py
ADD casestudy.ipynb /work/casestudy.ipynb
ADD AllData.pkl /work/images/preprocessed/


# Setup jupyter notebook as the default command
# This means that jupyter notebook is launched by default when doing `docker run`.
# Options:
#   --ip=0.0.0.0 bind on all interfaces (otherwise we cannot connect to it)
#   --allow-root force jupyter notebook to start even if we run as root inside the container
#   --NotebookApp.default_url=/notebooks/casestudy.ipynb Open the notebook by default
CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.default_url=/notebooks/casestudy.ipynb" ]

# Declare port 8888
EXPOSE 8888

