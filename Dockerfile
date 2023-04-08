FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./algo.py /code/algo.py

COPY ./configs.py /code/configs.py

COPY ./main.py /code/main.py

COPY ./model.py /code/model.py

COPY ./router.py /code/router.py

COPY ./mongodb.py /code/mongodb.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]