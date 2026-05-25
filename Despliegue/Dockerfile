FROM python:3.11-slim

WORKDIR /opt/app

COPY . .

RUN pip install --upgrade pip
RUN pip install pandas numpy joblib dash plotly flask gunicorn
RUN pip install scikit-learn==1.8.0
RUN pip install tf-nightly keras-nightly

EXPOSE 8050

CMD ["python", "despliegue.py"]