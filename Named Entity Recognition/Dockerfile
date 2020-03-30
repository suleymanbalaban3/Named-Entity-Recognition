FROM python:3

RUN mkdir /model
ADD NerWebApi.py /
ADD model /model/

RUN pip install spacy
RUN pip install flask

CMD [ "python", "./NerWebApi.py" ]