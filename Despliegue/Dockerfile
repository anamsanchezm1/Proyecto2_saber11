FROM python:3.10-slim

WORKDIR /opt/app

COPY . .

RUN pip install --upgrade pip
RUN pip install pandas numpy scikit-learn joblib dash plotly flask gunicorn tensorflow
RUN pip install tensorflow==2.20.0 keras==3.12.0

EXPOSE 8050

CMD ["python", "despliegue.py"]
