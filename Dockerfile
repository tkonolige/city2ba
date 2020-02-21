FROM ubuntu:latest
RUN apt-get update && apt-get install curl -y && apt-get clean
ADD https://github.com/embree/embree/releases/download/v3.8.0/embree-3.8.0.x86_64.linux.tar.gz /
RUN tar xvf embree-3.8.0.x86_64.linux.tar.gz
RUN mv embree-3.8.0.x86_64.linux/lib/* /usr/lib
RUN mv embree-3.8.0.x86_64.linux/include/* /usr/include
RUN rm -r embree-3.8.0.x86_64.linux embree-3.8.0.x86_64.linux.tar.gz
