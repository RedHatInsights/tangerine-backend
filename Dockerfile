#----------------------- base -----------------------
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest

ENV LC_ALL=C.utf8
ENV LANG=C.utf8
ENV PYTHONUNBUFFERED=1 
ENV PYTHONIOENCODING=UTF-8
ENV PIP_NO_CACHE_DIR=1

USER root

ENV APP_ROOT=/opt/app-root/src
WORKDIR $APP_ROOT

# install postgresql from centos if not building on RHEL host
RUN ON_RHEL=$(microdnf repolist --enabled | grep rhel-9) ; \
    if [ -z "$ON_RHEL" ] ; then \
        rpm -Uvh http://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os/Packages/centos-stream-repos-9.0-26.el9.noarch.rpm \
                 http://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os/Packages/centos-gpg-keys-9.0-26.el9.noarch.rpm && \
        sed -i 's/^\(enabled.*\)/\1\npriority=200/;' /etc/yum.repos.d/centos*.repo ; \
    fi

RUN microdnf -y module enable postgresql:16 && \
    microdnf -y upgrade && \
    microdnf -y install --setopt=install_weak_deps=0 --setopt=tsflags=nodocs python311 python3.11-pip which postgresql libpq && \
    rpm -qa | sort > packages-before-devel-install.txt && \
    microdnf -y install --setopt=install_weak_deps=0 --setopt=tsflags=nodocs python3.11-devel gcc make libpq-devel && \
    rpm -qa | sort > packages-after-devel-install.txt

COPY Pipfile .
COPY Pipfile.lock .
RUN python3.11 -m venv .venv && \
    source .venv/bin/activate && \
    python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install pipenv && \
    pipenv install --system

# remove devel packages that may have only been necessary for psycopg to compile
RUN microdnf remove -y $( comm -13 packages-before-devel-install.txt packages-after-devel-install.txt ) && \
    rm packages-before-devel-install.txt packages-after-devel-install.txt && \
    microdnf clean all && \
    rm -rf /mnt/rootfs/var/cache/* /mnt/rootfs/var/log/dnf* /mnt/rootfs/var/log/yum.*

USER 1001

ADD connectors ./connectors
ADD resources ./resources
ADD utils ./utils
ADD main.py .

EXPOSE 5000

ENV PATH="$APP_ROOT/.venv/bin:$PATH"
CMD ["python", "./main.py"]

