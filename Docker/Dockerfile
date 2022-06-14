FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Create new non-root user
ARG USERNAME=mluser
ARG USERID=1000
RUN useradd --system --create-home --shell /bin/bash --uid $USERID $USERNAME \
 # To ensure proper permissions for mapped volume, we need to create its folder
 # and set permissions during the image build
 && mkdir /home/$USERNAME/.cache && chown -R $USERNAME /home/$USERNAME/.cache 

COPY requirements.txt /tmp/requirements.txt 
RUN pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

USER $USERNAME
COPY --chown=$USERNAME ./app /home/$USERNAME/app
WORKDIR /home/$USERNAME/app

# Optional: switch back to dialog (for any ad-hoc use of apt-get on running container)
ENV DEBIAN_FRONTEND=dialog

# Optional: ensure explicit language and locale configuration
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENTRYPOINT ["python", "nlp.py"]
