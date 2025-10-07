######## Option 1: With Debug, but DOESN'T work with Coursera 

# FROM continuumio/miniconda3@sha256:d601a04ea48fd45e60808c7072243d33703d29434d2067816b7f26b0705d889a

# RUN apk update && apk add libstdc++

########

######## Option 2: No Debug, but works with Coursera

FROM continuumio/miniconda3@sha256:e7f31b55e0006cf2a545cc5b2e60c9bb6cc24027d1cea01c61e990f34f44d387

RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

########

COPY requirements.txt .

RUN pip install -r requirements.txt &&     rm requirements.txt

RUN mkdir /grader && \ 
mkdir /grader/submission

COPY .conf /grader/.conf
COPY data/ /grader/data/
COPY grader_utils.py /grader/grader_utils.py
COPY entry.py /grader/entry.py
COPY grader.py /grader/grader.py

RUN chmod a+rwx /grader/

WORKDIR /grader/

ENTRYPOINT ["python", "entry.py"]
